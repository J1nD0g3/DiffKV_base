"""Tests for Qwen3 Think/Reasoning mode in DiffKV.

Verifies:
1. Think mode flag and token constants in sparse_qwen3
2. Prompt template construction (build_think_prompt)
3. SamplingParams compatibility for Think mode
4. LongBench data loading and truncation logic
"""
import sys
import os
import pytest

# Ensure DiffKV root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ─── Test 1: Think mode constants and model config ───────────────────────────

class TestThinkModeConstants:
    """Verify that Think-mode constants are correctly defined."""

    def test_think_tokens_defined(self):
        from vllm.model_executor.models.sparse_qwen3 import (
            THINK_START_TOKEN,
            THINK_END_TOKEN,
        )
        assert THINK_START_TOKEN == "<think>"
        assert THINK_END_TOKEN == "</think>"

    def test_think_tokens_are_valid_xml_tags(self):
        from vllm.model_executor.models.sparse_qwen3 import (
            THINK_START_TOKEN,
            THINK_END_TOKEN,
        )
        assert THINK_START_TOKEN.startswith("<")
        assert THINK_START_TOKEN.endswith(">")
        assert THINK_END_TOKEN.startswith("</")
        assert THINK_END_TOKEN.endswith(">")
        # End token should be the closing tag of start token
        tag_name = THINK_START_TOKEN.strip("<>")
        assert THINK_END_TOKEN == f"</{tag_name}>"


# ─── Test 2: build_think_prompt ──────────────────────────────────────────────

class TestBuildThinkPrompt:
    """Verify the Think prompt template builder."""

    def test_prompt_has_think_tag(self):
        from vllm.model_executor.models.sparse_qwen3 import (
            Qwen3ForCausalLM,
            THINK_START_TOKEN,
        )
        prompt = Qwen3ForCausalLM.build_think_prompt("What is 2+2?")
        assert THINK_START_TOKEN in prompt
        assert prompt.endswith(f"{THINK_START_TOKEN}\n")

    def test_prompt_has_step_by_step_instruction(self):
        from vllm.model_executor.models.sparse_qwen3 import Qwen3ForCausalLM
        prompt = Qwen3ForCausalLM.build_think_prompt("Solve x^2 = 4")
        assert "step by step" in prompt.lower()
        assert "\\boxed{}" in prompt

    def test_prompt_preserves_original(self):
        from vllm.model_executor.models.sparse_qwen3 import Qwen3ForCausalLM
        original = "Calculate the integral of x dx"
        prompt = Qwen3ForCausalLM.build_think_prompt(original)
        assert prompt.startswith(original)

    def test_empty_prompt(self):
        from vllm.model_executor.models.sparse_qwen3 import Qwen3ForCausalLM
        prompt = Qwen3ForCausalLM.build_think_prompt("")
        assert "<think>" in prompt


# ─── Test 3: SamplingParams compatibility ────────────────────────────────────

class TestThinkSamplingParams:
    """Verify that Think-mode SamplingParams are valid."""

    def test_default_params_creation(self):
        from vllm.model_executor.models.sparse_qwen3 import Qwen3ForCausalLM
        params = Qwen3ForCausalLM.get_think_sampling_params()
        assert params is not None

    def test_stop_token_is_think_end(self):
        from vllm.model_executor.models.sparse_qwen3 import (
            Qwen3ForCausalLM,
            THINK_END_TOKEN,
        )
        params = Qwen3ForCausalLM.get_think_sampling_params()
        assert THINK_END_TOKEN in params.stop

    def test_qwq_style_temperature(self):
        from vllm.model_executor.models.sparse_qwen3 import Qwen3ForCausalLM
        params = Qwen3ForCausalLM.get_think_sampling_params()
        # QwQ uses temperature=0.6
        assert abs(params.temperature - 0.6) < 1e-6

    def test_custom_max_tokens(self):
        from vllm.model_executor.models.sparse_qwen3 import Qwen3ForCausalLM
        params = Qwen3ForCausalLM.get_think_sampling_params(max_tokens=16384)
        assert params.max_tokens == 16384

    def test_custom_temperature(self):
        from vllm.model_executor.models.sparse_qwen3 import Qwen3ForCausalLM
        params = Qwen3ForCausalLM.get_think_sampling_params(temperature=0.8)
        assert abs(params.temperature - 0.8) < 1e-6

    def test_params_are_not_greedy(self):
        """Think mode should use random sampling, not greedy."""
        from vllm.model_executor.models.sparse_qwen3 import Qwen3ForCausalLM
        params = Qwen3ForCausalLM.get_think_sampling_params()
        assert params.temperature > 0.0

    def test_top_p_top_k_set(self):
        from vllm.model_executor.models.sparse_qwen3 import Qwen3ForCausalLM
        params = Qwen3ForCausalLM.get_think_sampling_params()
        assert params.top_p == 0.95
        assert params.top_k == 40


# ─── Test 4: LongBench loader ───────────────────────────────────────────────

class TestLongBenchSubsets:
    """Verify LongBench subset catalogs in benchmark_throughput."""

    def test_en_subsets_exist(self):
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks"))
        from benchmark_throughput import LONGBENCH_SUBSETS_EN
        assert len(LONGBENCH_SUBSETS_EN) > 0
        assert "qasper" in LONGBENCH_SUBSETS_EN
        assert "hotpotqa" in LONGBENCH_SUBSETS_EN

    def test_all_subsets_exist(self):
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks"))
        from benchmark_throughput import LONGBENCH_SUBSETS_ALL
        assert len(LONGBENCH_SUBSETS_ALL) > 0
        # ALL should be a superset of EN
        from benchmark_throughput import LONGBENCH_SUBSETS_EN
        for s in LONGBENCH_SUBSETS_EN:
            assert s in LONGBENCH_SUBSETS_ALL

    def test_en_subsets_are_all_english(self):
        """EN subset list should not contain Chinese-only subsets."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks"))
        from benchmark_throughput import LONGBENCH_SUBSETS_EN
        zh_subsets = {"multifieldqa_zh", "dureader", "vcsum", "lsht",
                      "passage_retrieval_zh"}
        for s in LONGBENCH_SUBSETS_EN:
            assert s not in zh_subsets, f"EN subset should not contain {s}"


# ─── Test 5: SparsePagedAttention is untouched ──────────────────────────────

class TestSparseAttentionIntegrity:
    """Verify that the DiffKV SparsePagedAttention path is preserved."""

    def test_attention_uses_sparse_paged(self):
        """Qwen3Attention in sparse_qwen3 must use SparsePagedAttention."""
        import inspect
        from vllm.model_executor.models.sparse_qwen3 import Qwen3Attention
        source = inspect.getsource(Qwen3Attention)
        assert "SparsePagedAttention" in source

    def test_attention_has_kv_buffer_size(self):
        """Qwen3Attention must accept kv_buffer_size parameter."""
        import inspect
        from vllm.model_executor.models.sparse_qwen3 import Qwen3Attention
        sig = inspect.signature(Qwen3Attention.__init__)
        assert "kv_buffer_size" in sig.parameters

    def test_attention_has_qk_norms(self):
        """Qwen3Attention must have q_norm and k_norm (QK normalization)."""
        import inspect
        from vllm.model_executor.models.sparse_qwen3 import Qwen3Attention
        source = inspect.getsource(Qwen3Attention.__init__)
        assert "q_norm" in source
        assert "k_norm" in source

    def test_model_has_quant_config(self):
        """Qwen3ForCausalLM must accept quant_config parameter."""
        import inspect
        from vllm.model_executor.models.sparse_qwen3 import Qwen3ForCausalLM
        sig = inspect.signature(Qwen3ForCausalLM.__init__)
        assert "quant_config" in sig.parameters


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

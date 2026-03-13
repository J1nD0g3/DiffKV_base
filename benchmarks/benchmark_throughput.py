"""Benchmark offline inference throughput.

Supports two dataset modes:
  --dataset math      : EleutherAI/hendrycks_math  (short context, legacy)
  --dataset longbench : THUDM/LongBench            (long context, default)

When --enable-thinking is set, prompts are wrapped with <think>\n for Qwen3
Think/Reasoning mode and sampling uses QwQ-style parameters.
"""
import argparse
import json
import pathlib
import time
from typing import List, Optional, Tuple

from transformers import AutoTokenizer
from tqdm import tqdm


# ─── LongBench subset catalog ────────────────────────────────────────────────
LONGBENCH_SUBSETS_EN = [
    "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa",
    "gov_report", "multi_news", "trec", "triviaqa", "samsum",
    "passage_count", "passage_retrieval_en", "lcc", "repobench-p",
]

LONGBENCH_SUBSETS_ALL = [
    "narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh",
    "hotpotqa", "2wikimqa", "musique", "dureader",
    "gov_report", "qmsum", "multi_news", "vcsum",
    "trec", "triviaqa", "samsum", "lsht",
    "passage_count", "passage_retrieval_en", "passage_retrieval_zh",
    "lcc", "repobench-p",
]


# ─── Dataset loaders ─────────────────────────────────────────────────────────

def sample_requests_math(
    tokenizer,
    num_requests,
    max_output_len,
    max_model_len: int = 32768,
    enable_thinking: bool = False,
):
    """Load EleutherAI/hendrycks_math and return (prompt, prompt_len, output_len) tuples.

    Uses the tokenizer's chat template to format prompts properly for Qwen3.
    The chat template natively supports enable_thinking (adds <think> tags
    when True, empty think block when False).

    Args:
        tokenizer: HuggingFace tokenizer for prompt length calculation.
        num_requests: Maximum number of requests to sample.
        max_output_len: Target maximum output length per request.
        max_model_len: Hard limit — prompts longer than this are truncated.
        enable_thinking: If True, enable Qwen3 think/reasoning mode.
    """
    from datasets import load_dataset, concatenate_datasets

    subjects = [
        'algebra', 'counting_and_probability', 'geometry',
        'intermediate_algebra', 'number_theory', 'prealgebra', 'precalculus',
    ]
    dataset = concatenate_datasets([
        load_dataset("EleutherAI/hendrycks_math", subject, split="test")
        for subject in subjects
    ])

    requests = []
    num_truncated = 0
    num_filtered = 0

    for data in dataset:
        user_prompt = data["problem"]

        # Apply chat template for proper Qwen3 formatting
        messages = [{"role": "user", "content": user_prompt}]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )

        output_len = max_output_len

        # Tokenize and enforce max_model_len
        input_ids = tokenizer(prompt, truncation=False).input_ids
        prompt_len = len(input_ids)

        budget = max_model_len - output_len

        if prompt_len > budget:
            if budget < 128:
                num_filtered += 1
                continue
            input_ids = input_ids[:budget]
            prompt = tokenizer.decode(input_ids, skip_special_tokens=False)
            prompt_len = budget
            num_truncated += 1

        requests.append((prompt, prompt_len, output_len))
        if len(requests) == num_requests:
            break

    if num_truncated > 0:
        print(f"[Math] Truncated {num_truncated} prompts to fit "
              f"max_model_len={max_model_len}")
    if num_filtered > 0:
        print(f"[Math] Filtered out {num_filtered} prompts "
              f"(too long even after truncation)")
    print(f"[Math] Loaded {len(requests)} requests "
          f"(subjects={len(subjects)}, thinking={enable_thinking})")

    return requests


def sample_requests_longbench(
    tokenizer,
    num_requests: int,
    max_output_len: int,
    max_model_len: int = 32768,
    subset: str = "all",
    sample_rate: int = 100,
    enable_thinking: bool = False,
):
    """Load THUDM/LongBench and return (prompt, prompt_len, output_len) tuples.

    Args:
        tokenizer: HuggingFace tokenizer for prompt length calculation.
        num_requests: Maximum number of requests to sample.
        max_output_len: Target maximum output length per request.
        max_model_len: Hard limit — prompts longer than this are truncated.
            Should match the model's max_position_embeddings.
        subset: A specific LongBench subset name (e.g., 'qasper', 'hotpotqa')
            or 'all' (all English subsets) or 'all_langs' (all subsets).
        sample_rate: Percentage of dataset to use (1–100).
        enable_thinking: If True, wrap prompts with <think> for reasoning.
    """
    from datasets import load_dataset, concatenate_datasets

    # ── Resolve subset list ──
    if subset == "all":
        subsets = LONGBENCH_SUBSETS_EN
    elif subset == "all_langs":
        subsets = LONGBENCH_SUBSETS_ALL
    else:
        assert subset in LONGBENCH_SUBSETS_ALL, (
            f"Unknown LongBench subset: {subset}. "
            f"Choose from: {LONGBENCH_SUBSETS_ALL}"
        )
        subsets = [subset]

    # ── Load prompt templates & max gen lengths ──
    _data_dir = pathlib.Path(__file__).parent.parent / "vllm" / "dataset"
    with open(_data_dir / "_longbench_prompts.json", "r") as f:
        prompt_formats = json.load(f)
    with open(_data_dir / "_longbench_max_len.json", "r") as f:
        subset_to_maxlen = json.load(f)

    # ── Load dataset(s) ──
    all_datasets = []
    for s in subsets:
        _ds = load_dataset("THUDM/LongBench", s, split="test", trust_remote_code=True)
        if sample_rate < 100:
            _ds = _ds.select(range(0, len(_ds), 100 // sample_rate))
        all_datasets.append(_ds)
    dataset = concatenate_datasets(all_datasets)

    # ── Build request tuples ──
    requests: List[Tuple[str, int, int]] = []
    num_truncated = 0
    num_filtered = 0

    for item in dataset:
        subset_name = item["dataset"]

        # Format prompt using LongBench template
        raw_prompt = prompt_formats[subset_name].format(**item)

        # Apply chat template for proper Qwen3 formatting
        messages = [{"role": "user", "content": raw_prompt}]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )

        # Determine output length
        if enable_thinking:
            output_len = max_output_len  # CoT needs full output budget
        else:
            # Use per-subset default if smaller than requested
            output_len = min(
                max_output_len,
                subset_to_maxlen.get(subset_name, max_output_len),
            )

        # Tokenize and enforce max_model_len
        input_ids = tokenizer(prompt, truncation=False).input_ids
        prompt_len = len(input_ids)

        budget = max_model_len - output_len

        if prompt_len > budget:
            # Try truncating the prompt to fit
            if budget < 128:
                # Too short to be meaningful after truncation — skip
                num_filtered += 1
                continue
            input_ids = input_ids[:budget]
            prompt = tokenizer.decode(input_ids, skip_special_tokens=False)
            prompt_len = budget
            num_truncated += 1

        requests.append((prompt, prompt_len, output_len))
        if len(requests) >= num_requests:
            break

    if num_truncated > 0:
        print(f"[LongBench] Truncated {num_truncated} prompts to fit "
              f"max_model_len={max_model_len}")
    if num_filtered > 0:
        print(f"[LongBench] Filtered out {num_filtered} prompts "
              f"(too long even after truncation)")
    print(f"[LongBench] Loaded {len(requests)} requests "
          f"(subsets={subset}, sample_rate={sample_rate}%)")

    return requests


def sample_requests(tokenizer, num_requests, max_output_len):
    """Legacy entrypoint — kept for backward compatibility."""
    return sample_requests_math(tokenizer, num_requests, max_output_len)


# ─── DiffKV engine runner ────────────────────────────────────────────────────

def run_diffkv(
    model: str,
    download_dir: str,
    requests: List[Tuple[str, int, int]],
    tensor_parallel_size: int,
    max_batch_size: int,
    quant_config: List[int],
    compress_config: List[float],
    gpu_memory_utilization: float,
    enable_thinking: bool = False,
) -> float:
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model,
        download_dir=download_dir,
        load_format='safetensors',
        tensor_parallel_size=tensor_parallel_size,
        max_num_seqs=max_batch_size,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
        dtype='float16',
        enforce_eager=True,
        kv_buffer_size=64,
    )

    # ── Choose sampling params ──
    if enable_thinking:
        from vllm.model_executor.models.sparse_qwen3 import (
            Qwen3ForCausalLM,
            THINK_END_TOKEN,
        )
        think_params = Qwen3ForCausalLM.get_think_sampling_params()

    # Stop token for chat model
    IM_END_TOKEN = "<|im_end|>"

    for prompt, _, max_output_len in requests:
        if enable_thinking:
            sampling_params = SamplingParams(
                n=1,
                temperature=think_params.temperature,
                top_p=think_params.top_p,
                top_k=think_params.top_k,
                min_p=think_params.min_p,
                max_tokens=max_output_len,
                stop=[THINK_END_TOKEN, IM_END_TOKEN],
                include_stop_str_in_output=True,
            )
        else:
            sampling_params = SamplingParams(
                n=1,
                temperature=0.0,
                use_beam_search=False,
                ignore_eos=False,
                max_tokens=max_output_len,
            )

        llm._add_request(
            prompt=prompt,
            prompt_token_ids=None,
            sampling_params=sampling_params,
            quant_configs=quant_config,
            quant_groups=[1, 1, 1, 1],
            compress_configs=compress_config,
        )

    start = time.perf_counter()
    outputs = llm._run_engine(use_tqdm=True)
    end = time.perf_counter()
    elapsed_time = end - start

    total_num_tokens = 0
    think_block_count = 0
    total_output_tokens = 0
    eos_stopped_count = 0
    length_stopped_count = 0
    for output in outputs:
        prompt_len = len(output.prompt_token_ids)
        output_text = output.outputs[0].text
        output_len = len(output.outputs[0].token_ids)
        finish_reason = output.outputs[0].finish_reason
        total_num_tokens += (prompt_len + output_len)
        total_output_tokens += output_len

        # Count Think blocks by checking for </think> in output text.
        # include_stop_str_in_output=True keeps the stop string in the text,
        # so this is an unambiguous signal that the model actually produced
        # a </think> token (not just EOS or max_tokens).
        if enable_thinking:
            if THINK_END_TOKEN in output_text:
                think_block_count += 1
            elif finish_reason == "stop":
                eos_stopped_count += 1
            elif finish_reason == "length":
                length_stopped_count += 1

    avg_output = total_output_tokens / len(outputs) if outputs else 0
    print(f"Throughput: {len(requests) / elapsed_time:.2f} requests/s, "
          f"{total_num_tokens / elapsed_time:.2f} tokens/s")
    print(f"Avg output tokens: {avg_output:.0f}")
    if enable_thinking:
        print(f"Think blocks detected: {think_block_count}/{len(outputs)} outputs")
        print(f"  - EOS stopped (no think block): {eos_stopped_count}")
        print(f"  - Max tokens reached: {length_stopped_count}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace):
    print(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # ── Load dataset ──
    if args.dataset == "longbench":
        requests = sample_requests_longbench(
            tokenizer,
            num_requests=args.num_requests,
            max_output_len=args.max_output_len,
            max_model_len=args.max_model_len,
            subset=args.longbench_subset,
            sample_rate=args.longbench_sample_rate,
            enable_thinking=args.enable_thinking,
        )
    else:
        requests = sample_requests_math(
            tokenizer,
            num_requests=args.num_requests,
            max_output_len=args.max_output_len,
            max_model_len=args.max_model_len,
            enable_thinking=args.enable_thinking,
        )

    quant_config = [args.kbits_high, args.vbits_high,
                    args.kbits_low, args.vbits_low]
    if args.kbits_high == args.kbits_low and args.vbits_high == args.vbits_low:
        quant_config = [args.kbits_high, args.vbits_high]

    compress_config = [args.kv_prune_thresh, args.kv_quant_thresh]

    run_diffkv(
        args.model, args.download_dir, requests,
        args.tensor_parallel_size, args.max_batch_size,
        quant_config, compress_config,
        args.gpu_memory_utilization,
        enable_thinking=args.enable_thinking,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark the throughput.")

    # ── Model ──
    parser.add_argument("--model", type=str, help="The model to use.")
    parser.add_argument("--download-dir", type=str,
                        help="The directory to download models from huggingface.")

    # ── Dataset ──
    parser.add_argument("--dataset", type=str, default="longbench",
                        choices=["math", "longbench"],
                        help="Dataset to benchmark (default: longbench).")
    parser.add_argument("--longbench-subset", type=str, default="all",
                        help="LongBench subset: 'all', 'all_langs', or a "
                             "specific name like 'qasper'.")
    parser.add_argument("--longbench-sample-rate", type=int, default=100,
                        help="Percentage of LongBench dataset to use (1–100).")

    # ── Sequence lengths ──
    parser.add_argument("--num-requests", type=int, default=512,
                        help="The number of requests to sample.")
    parser.add_argument("--max-output-len", type=int, default=4096,
                        help="The maximum output length.")
    parser.add_argument("--max-model-len", type=int, default=32768,
                        help="Max context length; prompts exceeding this "
                             "are truncated.")

    # ── Think mode ──
    parser.add_argument("--enable-thinking", action="store_true",
                        help="Enable Qwen3 Think/CoT mode (injects <think> "
                             "tags and uses QwQ sampling).")

    # ── Execution ──
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-batch-size", type=int, default=256)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)

    # ── DiffKV compression ──
    parser.add_argument("--kbits-high", type=int, default=8)
    parser.add_argument("--vbits-high", type=int, default=4)
    parser.add_argument("--kbits-low", type=int, default=4)
    parser.add_argument("--vbits-low", type=int, default=2)
    parser.add_argument("--kv-prune-thresh", type=float)
    parser.add_argument("--kv-quant-thresh", type=float)

    args = parser.parse_args()
    main(args)

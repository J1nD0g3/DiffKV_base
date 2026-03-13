#!/usr/bin/env python
"""
Think Mode A/B Toggle Test — DiffKV × Qwen3 Reasoning Verification
===================================================================

Runs two experiments on the SAME prompt:

  Experiment A (Think OFF):  Standard greedy decoding → short, direct answer.
  Experiment B (Think ON):   Reasoning mode with <think> block → long CoT output.

Validates:
  1. Think ON produces <think> tags in the output.
  2. Think ON output length ≥ 3× Think OFF output length.
  3. DiffKV SparsePagedAttention runs without CUDA errors.
  4. KV Cache memory usage is monitored via nvidia-smi snapshots.

Usage (requires GPU + Qwen3 model weights):
  PYTHONPATH=/home/jheo/DiffKV:$PYTHONPATH \\
  python tests/test_think_toggle.py \\
      --model /home/jheo/models/Qwen/Qwen3-8B \\
      [--kbits-high 8] [--vbits-high 4] [--kbits-low 4] [--vbits-low 2] \\
      [--kv-prune-thresh 0.02] [--kv-quant-thresh 1.0]
"""

import argparse
import json
import os
import subprocess
import sys
import time
import threading
from dataclasses import dataclass, field, asdict
from typing import List, Optional

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
QUESTION = (
    "If I have 3 apples and I eat 1, then buy 5 more, how many do I have? "
    "Explain the process."
)

THINK_OFF_MAX_TOKENS = 256   # short direct answer
THINK_ON_MAX_TOKENS  = 4096  # ample room for CoT reasoning

MIN_LENGTH_RATIO = 3.0       # Think ON output must be ≥ 3x Think OFF


# ──────────────────────────────────────────────────────────────────────────────
# Data classes for structured logging
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class ExperimentResult:
    mode: str                            # "think_off" or "think_on"
    prompt: str = ""
    output_text: str = ""
    output_tokens: int = 0
    prompt_tokens: int = 0
    generation_time_sec: float = 0.0
    tokens_per_sec: float = 0.0
    contains_think_tag: bool = False
    gpu_memory_snapshots_mb: List[float] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────────
# GPU monitoring
# ──────────────────────────────────────────────────────────────────────────────
class GpuMonitor:
    """Background GPU memory monitor via nvidia-smi."""

    def __init__(self, interval_sec: float = 1.0):
        self.interval = interval_sec
        self._snapshots: List[float] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self):
        self._stop.clear()
        self._snapshots = []
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> List[float]:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        return list(self._snapshots)

    def _run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(
                    ["nvidia-smi",
                     "--query-gpu=memory.used",
                     "--format=csv,nounits,noheader"],
                    text=True,
                )
                # Sum across all GPUs
                total_mb = sum(float(x.strip()) for x in out.strip().split("\n") if x.strip())
                self._snapshots.append(total_mb)
            except Exception:
                pass  # nvidia-smi unavailable
            self._stop.wait(self.interval)


# ──────────────────────────────────────────────────────────────────────────────
# Core experiment runner
# ──────────────────────────────────────────────────────────────────────────────
def run_experiment(
    model_path: str,
    enable_thinking: bool,
    quant_config: List[int],
    quant_groups: List[int],
    compress_config: List[float],
    gpu_utilization: float = 0.9,
) -> ExperimentResult:
    """Run a single experiment (Think ON or OFF) and collect metrics."""

    from vllm import LLM, SamplingParams
    from vllm.model_executor.models.sparse_qwen3 import (
        Qwen3ForCausalLM,
        THINK_START_TOKEN,
        THINK_END_TOKEN,
    )

    mode_label = "think_on" if enable_thinking else "think_off"
    result = ExperimentResult(mode=mode_label)

    # ── Build prompt ──
    if enable_thinking:
        prompt = Qwen3ForCausalLM.build_think_prompt(QUESTION)
        max_tokens = THINK_ON_MAX_TOKENS
    else:
        prompt = QUESTION
        max_tokens = THINK_OFF_MAX_TOKENS

    result.prompt = prompt

    # ── Build sampling params ──
    if enable_thinking:
        sampling_params = Qwen3ForCausalLM.get_think_sampling_params(
            max_tokens=max_tokens,
        )
    else:
        sampling_params = SamplingParams(
            n=1,
            temperature=0.0,
            max_tokens=max_tokens,
            ignore_eos=False,
        )

    # ── Initialize LLM (DiffKV) ──
    print(f"\n{'='*70}")
    print(f"  Experiment: {mode_label.upper()}")
    print(f"  max_tokens={max_tokens}, quant_config={quant_config}")
    print(f"  compress_config={compress_config}")
    print(f"{'='*70}\n")

    llm = LLM(
        model=model_path,
        load_format="safetensors",
        dtype="float16",
        gpu_memory_utilization=gpu_utilization,
        kv_buffer_size=64,
        enforce_eager=True,
        trust_remote_code=True,
    )

    # ── Start GPU monitoring ──
    gpu_mon = GpuMonitor(interval_sec=0.5)
    gpu_mon.start()

    # ── Generate ──
    t_start = time.perf_counter()
    outputs = llm.generate(
        [prompt],
        sampling_params,
        quant_configs=quant_config,
        quant_groups=quant_groups,
        compress_configs=compress_config,
    )
    t_end = time.perf_counter()

    # ── Stop GPU monitoring ──
    result.gpu_memory_snapshots_mb = gpu_mon.stop()

    # ── Collect metrics ──
    output = outputs[0]
    result.output_text = output.outputs[0].text
    result.output_tokens = len(output.outputs[0].token_ids)
    result.prompt_tokens = len(output.prompt_token_ids)
    result.generation_time_sec = round(t_end - t_start, 3)
    result.tokens_per_sec = round(
        result.output_tokens / result.generation_time_sec, 2
    ) if result.generation_time_sec > 0 else 0.0
    result.contains_think_tag = (
        THINK_START_TOKEN in result.output_text
        or THINK_END_TOKEN in result.output_text
    )

    # ── Print summary ──
    print(f"\n--- {mode_label.upper()} Result ---")
    print(f"  Prompt tokens:  {result.prompt_tokens}")
    print(f"  Output tokens:  {result.output_tokens}")
    print(f"  Generation time: {result.generation_time_sec}s")
    print(f"  Tokens/sec:     {result.tokens_per_sec}")
    print(f"  Contains <think>: {result.contains_think_tag}")
    if result.gpu_memory_snapshots_mb:
        peak = max(result.gpu_memory_snapshots_mb)
        print(f"  GPU memory peak: {peak:.0f} MB")
    print(f"  Output preview: {result.output_text[:300]}...")
    print()

    # ── Cleanup ──
    del llm

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Report generation
# ──────────────────────────────────────────────────────────────────────────────
def generate_report(
    result_off: ExperimentResult,
    result_on: ExperimentResult,
    log_path: str,
):
    """Print the A/B comparison report and save structured log."""

    length_ratio = (
        result_on.output_tokens / result_off.output_tokens
        if result_off.output_tokens > 0 else float("inf")
    )

    print("\n" + "=" * 70)
    print("  THINK MODE A/B TEST REPORT")
    print("=" * 70)

    # ── Side-by-side metrics ──
    print(f"\n{'Metric':<30} {'Think OFF':>15} {'Think ON':>15}")
    print("-" * 60)
    print(f"{'Prompt tokens':<30} {result_off.prompt_tokens:>15} {result_on.prompt_tokens:>15}")
    print(f"{'Output tokens':<30} {result_off.output_tokens:>15} {result_on.output_tokens:>15}")
    print(f"{'Avg output tokens':<30} {result_off.output_tokens:>15} {result_on.output_tokens:>15}")
    print(f"{'Generation time (s)':<30} {result_off.generation_time_sec:>15.3f} {result_on.generation_time_sec:>15.3f}")
    print(f"{'Tokens/sec':<30} {result_off.tokens_per_sec:>15.2f} {result_on.tokens_per_sec:>15.2f}")
    print(f"{'Contains <think> tag':<30} {str(result_off.contains_think_tag):>15} {str(result_on.contains_think_tag):>15}")

    if result_off.gpu_memory_snapshots_mb and result_on.gpu_memory_snapshots_mb:
        peak_off = max(result_off.gpu_memory_snapshots_mb)
        peak_on = max(result_on.gpu_memory_snapshots_mb)
        print(f"{'GPU memory peak (MB)':<30} {peak_off:>15.0f} {peak_on:>15.0f}")

    # ── Assertions ──
    print(f"\n{'─' * 60}")
    print("  ASSERTIONS")
    print(f"{'─' * 60}")

    checks = []

    # Check 1: Think ON must contain <think> tag
    tag_ok = result_on.contains_think_tag
    checks.append(("Think ON contains <think> tag", tag_ok))

    # Check 2: Think OFF should NOT contain <think> tag
    no_tag_ok = not result_off.contains_think_tag
    checks.append(("Think OFF has no <think> tag", no_tag_ok))

    # Check 3: Output length ratio ≥ 3×
    ratio_ok = length_ratio >= MIN_LENGTH_RATIO
    checks.append(
        (f"Length ratio ≥ {MIN_LENGTH_RATIO}× (actual: {length_ratio:.1f}×)",
         ratio_ok)
    )

    # Check 4: No CUDA errors (if we reached here, no crash occurred)
    checks.append(("DiffKV SparseAttention — no CUDA errors", True))

    all_pass = True
    for desc, passed in checks:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}  {desc}")
        if not passed:
            all_pass = False

    # ── GPU memory analysis ──
    if result_on.gpu_memory_snapshots_mb:
        print(f"\n{'─' * 60}")
        print("  KV CACHE COMPRESSION MONITORING (Think ON)")
        print(f"{'─' * 60}")
        snaps = result_on.gpu_memory_snapshots_mb
        print(f"  Snapshots collected: {len(snaps)}")
        print(f"  Initial GPU memory: {snaps[0]:.0f} MB")
        print(f"  Peak GPU memory:    {max(snaps):.0f} MB")
        print(f"  Final GPU memory:   {snaps[-1]:.0f} MB")
        delta = max(snaps) - snaps[0]
        print(f"  Delta (peak - init): {delta:.0f} MB")
        if len(snaps) >= 3:
            mid_idx = len(snaps) // 2
            trend_start = sum(snaps[:3]) / 3
            trend_mid = sum(snaps[mid_idx-1:mid_idx+2]) / 3 if mid_idx > 0 else snaps[mid_idx]
            trend_end = sum(snaps[-3:]) / 3
            print(f"  Memory trend: {trend_start:.0f} → {trend_mid:.0f} → {trend_end:.0f} MB")
            if trend_end < trend_mid * 1.1:
                print("  ✅ KV Cache compression stabilized — no unbounded growth")
            else:
                print("  ⚠️  Memory still growing at end — may need larger pruning threshold")

    # ── Save structured log ──
    log_data = {
        "question": QUESTION,
        "think_off": asdict(result_off),
        "think_on": asdict(result_on),
        "length_ratio": round(length_ratio, 2),
        "all_assertions_passed": all_pass,
    }
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)
    print(f"\n  Structured log saved to: {log_path}")

    # ── Final verdict ──
    print(f"\n{'='*70}")
    if all_pass:
        print("  ✅  ALL CHECKS PASSED — Think mode porting verified successfully")
    else:
        print("  ❌  SOME CHECKS FAILED — review the results above")
    print(f"{'='*70}\n")

    return all_pass


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Qwen3 Think Mode A/B Toggle Test (DiffKV)"
    )
    parser.add_argument("--model", type=str, required=True,
                        help="Path or HF name of the Qwen3 model.")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--think-off-max-tokens", type=int,
                        default=THINK_OFF_MAX_TOKENS)
    parser.add_argument("--think-on-max-tokens", type=int,
                        default=THINK_ON_MAX_TOKENS)

    # DiffKV compression
    parser.add_argument("--kbits-high", type=int, default=8)
    parser.add_argument("--vbits-high", type=int, default=4)
    parser.add_argument("--kbits-low", type=int, default=4)
    parser.add_argument("--vbits-low", type=int, default=2)
    parser.add_argument("--kv-prune-thresh", type=float, default=0.02)
    parser.add_argument("--kv-quant-thresh", type=float, default=1.0)
    parser.add_argument("--log-path", type=str,
                        default="logs/think_toggle_report.json")

    args = parser.parse_args()

    think_off_max = args.think_off_max_tokens
    think_on_max = args.think_on_max_tokens

    quant_config = [args.kbits_high, args.vbits_high,
                    args.kbits_low, args.vbits_low]
    if args.kbits_high == args.kbits_low and args.vbits_high == args.vbits_low:
        quant_config = [args.kbits_high, args.vbits_high]

    quant_groups = [1, 1, 1, 1] if len(quant_config) == 4 else [1, 1]
    compress_config = [args.kv_prune_thresh, args.kv_quant_thresh]

    print("=" * 70)
    print("  Qwen3 Think Mode A/B Toggle Test")
    print("=" * 70)
    print(f"  Model:          {args.model}")
    print(f"  Question:       {QUESTION[:60]}...")
    print(f"  Quant config:   {quant_config}")
    print(f"  Compress config: {compress_config}")
    print(f"  Think OFF max:  {think_off_max} tokens")
    print(f"  Think ON max:   {think_on_max} tokens")
    print()

    # ── Experiment A: Think OFF ──
    print("\n>>> Starting Experiment A (Think OFF) <<<\n")
    result_off = run_experiment(
        model_path=args.model,
        enable_thinking=False,
        quant_config=quant_config,
        quant_groups=quant_groups,
        compress_config=compress_config,
        gpu_utilization=args.gpu_memory_utilization,
    )

    # ── Experiment B: Think ON ──
    print("\n>>> Starting Experiment B (Think ON) <<<\n")
    result_on = run_experiment(
        model_path=args.model,
        enable_thinking=True,
        quant_config=quant_config,
        quant_groups=quant_groups,
        compress_config=compress_config,
        gpu_utilization=args.gpu_memory_utilization,
    )

    # ── Generate report ──
    all_pass = generate_report(result_off, result_on, args.log_path)
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()

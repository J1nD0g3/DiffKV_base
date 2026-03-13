# Set PYTHONPATH to the project's root directory
export PYTHONPATH=/home/jheo/DiffKV:$PYTHONPATH

# Set the HuggingFace model download directory
export HF_DOWNLOAD_DIR=/home/jheo/models

# Need 4 GPUs to run Llama3-70B
# export CUDA_VISIBLE_DEVICES=0,1,2,3
#export CUDA_VISIBLE_DEVICES=0


# ═══════════════════════════════════════════════════════════════════════════════
#  Legacy: Math dataset benchmarks (hendrycks_math)
# ═══════════════════════════════════════════════════════════════════════════════

# Llama3-8B
# python benchmark_throughput.py \
#     --model /home/jheo/models/Meta-Llama-3-8B-Instruct \
#     --download-dir $HF_DOWNLOAD_DIR \
#     --dataset math \
#     --num-requests 512 \
#     --max-output-len 4096 \
#     --kbits-high 8 \
#     --vbits-high 4 \
#     --kbits-low 4 \
#     --vbits-low 2 \
#     --kv-prune-thresh 0.02 \
#     --kv-quant-thresh 1.0 \
#     --max-batch-size 128 > ../logs/llama3_8b.log

# # Qwen2.5-7B
# python benchmark_throughput.py \
#     --model /home/jheo/models/Qwen/Qwen2.5-7B-Instruct \
#     --download-dir $HF_DOWNLOAD_DIR \
#     --dataset math \
#     --num-requests 512 \
#     --max-output-len 4096 \
#     --kbits-high 8 \
#     --vbits-high 4 \
#     --kbits-low 4 \
#     --vbits-low 2 \
#     --kv-prune-thresh 0.04 \
#     --kv-quant-thresh 0.04 \
#     --max-batch-size 128 > ../logs/qwen2.5_7b.log

# # Llama3-70B
# python benchmark_throughput.py \
#     --model /home/jheo/models/Meta-Llama-3-70B-Instruct \
#     --download-dir $HF_DOWNLOAD_DIR \
#     --dataset math \
#     --tensor-parallel-size 4 \
#     --num-requests 512 \
#     --max-output-len 4096 \
#     --kbits-high 8 \
#     --vbits-high 4 \
#     --kbits-low 4 \
#     --vbits-low 2 \
#     --kv-prune-thresh 0.0 \
#     --kv-quant-thresh 1.0 \
#     --max-batch-size 64 > ../logs/llama3_70b.log

# # Qwen2.5-32B
# python benchmark_throughput.py \
#     --model /home/jheo/models/Qwen/Qwen2.5-32B-Instruct \
#     --download-dir $HF_DOWNLOAD_DIR \
#     --dataset math \
#     --tensor-parallel-size 2 \
#     --num-requests 512 \
#     --max-output-len 8192 \
#     --kbits-high 8 \
#     --vbits-high 4 \
#     --kbits-low 4 \
#     --vbits-low 2 \
#     --kv-prune-thresh 0.0 \
#     --kv-quant-thresh 3.0 \
#     --max-batch-size 32 > ../logs/qwen2.5_32b.log

# # QwQ-32B
# python benchmark_throughput.py \
#     --model /home/jheo/models/Qwen/QwQ-32B \
#     --download-dir $HF_DOWNLOAD_DIR \
#     --dataset math \
#     --tensor-parallel-size 2 \
#     --num-requests 512 \
#     --max-output-len 16384 \
#     --kbits-high 8 \
#     --vbits-high 4 \
#     --kbits-low 4 \
#     --vbits-low 2 \
#     --kv-prune-thresh 0.0 \
#     --kv-quant-thresh 3.0 \
#     --max-batch-size 16 > ../logs/qwq_32b.log


# ═══════════════════════════════════════════════════════════════════════════════
#  Qwen3 Think Mode + LongBench (long-context evaluation)
# ═══════════════════════════════════════════════════════════════════════════════
#
#  Guidelines for Qwen3 Reasoning + DiffKV compression:
#
#  • --max-output-len:
#      Reasoning models produce extensive CoT output.
#      Set ≥ 4096 (8192+ recommended for complex tasks like gov_report).
#
#  • --max-model-len:
#      LongBench prompts can exceed 20K tokens. Set this to the model's
#      max_position_embeddings (default: 32768 for Qwen3).
#
#  • --kv-prune-thresh:
#      Controls attention-score-based KV pruning. For long-context:
#        - 0.01: aggressive pruning — best throughput, slight quality loss
#        - 0.02: balanced (recommended for LongBench)
#        - 0.05: conservative — retains more context, slower
#
#  • --kv-quant-thresh:
#      Controls quantization threshold for KV cache. For long-context:
#        - 0.5:  aggressive — max compression, good for 32K+ inputs
#        - 1.0:  balanced (recommended starting point)
#        - 3.0:  conservative — minimal quality loss
#
#  • Quantization bits:
#      Long-context benefits from asymmetric quantization:
#        High-precision: kbits=8, vbits=4  (important/recent tokens)
#        Low-precision:  kbits=4, vbits=2  (older/pruned tokens)
# ═══════════════════════════════════════════════════════════════════════════════

# ── Qwen3-8B + LongBench (Think mode, single GPU) ──
# Balanced throughput & quality for 1-GPU setups.
# python benchmark_throughput.py \
#     --model /home/jheo/models/Qwen3-8B \
#     --download-dir $HF_DOWNLOAD_DIR \
#     --dataset longbench \
#     --longbench-subset all \
#     --longbench-sample-rate 100 \
#     --enable-thinking \
#     --num-requests 128 \
#     --max-output-len 8192 \
#     --max-model-len 32768 \
#     --kbits-high 8 \
#     --vbits-high 4 \
#     --kbits-low 4 \
#     --vbits-low 2 \
#     --kv-prune-thresh 0.02 \
#     --kv-quant-thresh 1.0 \
#     --max-batch-size 32 \
#     --gpu-memory-utilization 0.9 > ../logs/qwen3_8b_longbench_think.log

# jheo - Custom after feedback 
python benchmark_throughput.py \
    --model /home/jheo/models/Qwen3-8B \
    --download-dir $HF_DOWNLOAD_DIR \
    --dataset math \
    --num-requests 5 \
    --max-output-len 16384 \
    --max-model-len 32768 \
    --kbits-high 8 \
    --vbits-high 4 \
    --kbits-low 4 \
    --vbits-low 2 \
    --kv-prune-thresh 0.02 \
    --kv-quant-thresh 1.0 \
    --max-batch-size 32 \
    --gpu-memory-utilization 0.9 > ../logs/qwen3_8b_math_unthink.log

# ── Qwen3-8B + LongBench (single subset, no thinking) ──
# Useful for debugging or subset-specific evaluation.
# python benchmark_throughput.py \
#     --model '/home/jheo/models/Qwen/Qwen3-8B' \
#     --download-dir $HF_DOWNLOAD_DIR \
#     --dataset longbench \
#     --longbench-subset qasper \
#     --num-requests 64 \
#     --max-output-len 4096 \
#     --max-model-len 32768 \
#     --kbits-high 8 \
#     --vbits-high 4 \
#     --kbits-low 4 \
#     --vbits-low 2 \
#     --kv-prune-thresh 0.02 \
#     --kv-quant-thresh 1.0 \
#     --max-batch-size 64 > ../logs/qwen3_8b_longbench_qasper.log

# ── Qwen3-32B + LongBench (Think mode, 2×GPU) ──
# Maximum-quality reasoning with aggressive DiffKV compression.
# export CUDA_VISIBLE_DEVICES=0,1
# python benchmark_throughput.py \
#     --model '/home/jheo/models/Qwen/Qwen3-32B' \
#     --download-dir $HF_DOWNLOAD_DIR \
#     --dataset longbench \
#     --longbench-subset all \
#     --longbench-sample-rate 50 \
#     --enable-thinking \
#     --tensor-parallel-size 2 \
#     --num-requests 64 \
#     --max-output-len 16384 \
#     --max-model-len 32768 \
#     --kbits-high 8 \
#     --vbits-high 4 \
#     --kbits-low 4 \
#     --vbits-low 2 \
#     --kv-prune-thresh 0.01 \
#     --kv-quant-thresh 0.5 \
#     --max-batch-size 16 \
#     --gpu-memory-utilization 0.9 > ../logs/qwen3_32b_longbench_think.log

# ── Qwen3-8B + LongBench (aggressive compression stress-test) ──
# Push DiffKV compression to the limit for ablation studies.
# python benchmark_throughput.py \
#     --model '/home/jheo/models/Qwen/Qwen3-8B' \
#     --download-dir $HF_DOWNLOAD_DIR \
#     --dataset longbench \
#     --longbench-subset hotpotqa \
#     --enable-thinking \
#     --num-requests 64 \
#     --max-output-len 4096 \
#     --max-model-len 32768 \
#     --kbits-high 8 \
#     --vbits-high 4 \
#     --kbits-low 4 \
#     --vbits-low 2 \
#     --kv-prune-thresh 0.01 \
#     --kv-quant-thresh 0.5 \
#     --max-batch-size 64 > ../logs/qwen3_8b_longbench_aggressive.log

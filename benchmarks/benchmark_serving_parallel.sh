#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
#  1단계: 탐색 단계 — Qwen3-8B Serving Saturation Point 탐색 (2-GPU 병렬)
#  각 GPU에 vLLM 서버를 띄우고 Request Rate를 분배하여 병렬 실행
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

# ── 환경 설정 ──
export PYTHONPATH=/workspace/DiffKV_base:${PYTHONPATH:-}
export HF_DOWNLOAD_DIR=/workspace/models

MODEL=/workspace/models/Qwen3-8B
BASE_PORT=8010
LOG_DIR=/workspace/DiffKV_base/log/serving/exploratory
BENCHMARK_DIR=/workspace/DiffKV_base/benchmarks
NUM_GPUS=2

rm -rf "$LOG_DIR"
mkdir -p "$LOG_DIR"

# ── Request Rate 분배 ──
# 7개 rate를 2개 GPU에 분배
ALL_RATES=(0.1 0.2 0.5 1.0 2.0 5.0 10.0)
NUM_REQUESTS=1000

# GPU별 rate 할당 (라운드 로빈)
declare -a GPU0_RATES GPU1_RATES
for i in "${!ALL_RATES[@]}"; do
    gpu=$((i % NUM_GPUS))
    eval "GPU${gpu}_RATES+=(${ALL_RATES[$i]})"
done

# ── 서버 PID 추적 및 종료 핸들러 ──
declare -a SERVER_PIDS=()
declare -a WORKER_PIDS=()

cleanup() {
    echo "[INFO] Cleaning up..."
    for pid in "${WORKER_PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    done
    for pid in "${SERVER_PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    done
    echo "[INFO] All processes stopped."
}
trap cleanup EXIT

# ── GPU별 워커 함수 ──
run_worker() {
    local gpu_id=$1
    shift
    local rates=("$@")
    local port=$((BASE_PORT + gpu_id))

    echo "[GPU${gpu_id}] Starting vLLM server on port ${port}..."
    CUDA_VISIBLE_DEVICES=$gpu_id python -m vllm.entrypoints.api_server \
        --model "$MODEL" \
        --download-dir "$HF_DOWNLOAD_DIR" \
        --max-model-len 32768 \
        --max-num-seqs 32 \
        --gpu-memory-utilization 0.9 \
        --enforce-eager \
        --trust-remote-code \
        --dtype float16 \
        --load-format safetensors \
        --kv-buffer-size 64 \
        --port "$port" \
        &> "$LOG_DIR/server_gpu${gpu_id}.log" &
    local server_pid=$!
    echo "[GPU${gpu_id}] Server PID=${server_pid}"

    # 서버 준비 대기 (최대 5분)
    local max_wait=300
    local elapsed=0
    while ! curl -s "http://localhost:${port}/health" > /dev/null 2>&1; do
        sleep 5
        elapsed=$((elapsed + 5))
        if [ $elapsed -ge $max_wait ]; then
            echo "[GPU${gpu_id}] ERROR: Server failed to start within ${max_wait}s"
            return 1
        fi
        echo "[GPU${gpu_id}] Waiting for server... (${elapsed}s)"
    done
    echo "[GPU${gpu_id}] Server is ready!"

    # 할당된 rate들 순차 실행
    for rate in "${rates[@]}"; do
        echo ""
        echo "[GPU${gpu_id}] ═══ Request Rate: ${rate} req/s | Num Requests: ${NUM_REQUESTS} ═══"

        python "$BENCHMARK_DIR/benchmark_serving.py" \
            --model "$MODEL" \
            --port "$port" \
            --request-rate "$rate" \
            --num-requests "$NUM_REQUESTS" \
            --max-output-len 16384 \
            --kbits-high 8 \
            --vbits-high 4 \
            --kbits-low 4 \
            --vbits-low 2 \
            --kv-prune-thresh 0.02 \
            --kv-quant-thresh 1.0 \
            --enable-thinking \
            2>&1 | tee "$LOG_DIR/rate_${rate}.log"

        echo "[GPU${gpu_id}] Rate=${rate} completed."
    done

    # 워커 완료 후 서버 종료
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
    echo "[GPU${gpu_id}] Done. Server stopped."
}

# ── 2개 GPU에 워커 병렬 실행 ──
echo "═══════════════════════════════════════════════════════════════"
echo "  Rate 분배:"
echo "    GPU0: ${GPU0_RATES[*]}"
echo "    GPU1: ${GPU1_RATES[*]}"
echo "═══════════════════════════════════════════════════════════════"

run_worker 0 "${GPU0_RATES[@]}" &
WORKER_PIDS+=($!)

run_worker 1 "${GPU1_RATES[@]}" &
WORKER_PIDS+=($!)

# ── 모든 워커 완료 대기 ──
echo "[INFO] Waiting for all GPU workers to finish..."
FAIL=0
for pid in "${WORKER_PIDS[@]}"; do
    wait "$pid" || FAIL=$((FAIL + 1))
done

echo ""
echo "═══════════════════════════════════════════════════════════════"
if [ $FAIL -eq 0 ]; then
    echo "  1단계 탐색 완료! 로그 위치: $LOG_DIR/"
else
    echo "  완료 (${FAIL}개 워커 실패). 로그 확인: $LOG_DIR/"
fi
echo "═══════════════════════════════════════════════════════════════"

#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
#  1단계: 탐색 단계 — Qwen3-8B Serving Saturation Point 탐색
#  Request Rate를 변화시키며 Throughput/Latency 변화 관찰
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

# ── 환경 설정 ──
export PYTHONPATH=/workspace/DiffKV_base:${PYTHONPATH:-}
export HF_DOWNLOAD_DIR=/workspace/models

MODEL=/workspace/models/Qwen3-8B
PORT=8000
LOG_DIR=/workspace/DiffKV_base/log/serving/exploratory
BENCHMARK_DIR=/workspace/DiffKV_base/benchmarks

rm -rf "$LOG_DIR"
mkdir -p "$LOG_DIR"

# ── 서버 종료 핸들러 ──
SERVER_PID=""
cleanup() {
    if [ -n "$SERVER_PID" ]; then
        echo "[INFO] Shutting down vLLM server (PID=$SERVER_PID)..."
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        echo "[INFO] Server stopped."
    fi
}
trap cleanup EXIT

# ── vLLM 서버 시작 (백그라운드) ──
echo "[INFO] Starting vLLM server with DiffKV engine..."
python -m vllm.entrypoints.api_server \
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
    --port "$PORT" \
    &> "$LOG_DIR/server.log" &
SERVER_PID=$!
echo "[INFO] Server PID=$SERVER_PID, waiting for health check..."

# ── 서버 준비 대기 (최대 5분) ──
MAX_WAIT=300
ELAPSED=0
while ! curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; do
    sleep 5
    ELAPSED=$((ELAPSED + 5))
    if [ $ELAPSED -ge $MAX_WAIT ]; then
        echo "[ERROR] Server failed to start within ${MAX_WAIT}s. Check $LOG_DIR/server.log"
        exit 1
    fi
    echo "[INFO] Waiting for server... (${ELAPSED}s)"
done
echo "[INFO] Server is ready!"

# ── 탐색 실험: Request Rate 변동 ──
REQUEST_RATES=(0.1 0.2 0.5 1.0 2.0 5.0 10.0)
NUM_REQUESTS=1000

for RATE in "${REQUEST_RATES[@]}"; do
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "  Request Rate: ${RATE} req/s | Num Requests: ${NUM_REQUESTS}"
    echo "═══════════════════════════════════════════════════════════════"

    python "$BENCHMARK_DIR/benchmark_serving.py" \
        --model "$MODEL" \
        --port "$PORT" \
        --request-rate "$RATE" \
        --num-requests "$NUM_REQUESTS" \
        --max-output-len 16384 \
        --kbits-high 8 \
        --vbits-high 4 \
        --kbits-low 4 \
        --vbits-low 2 \
        --kv-prune-thresh 0.02 \
        --kv-quant-thresh 1.0 \
        --enable-thinking \
        2>&1 | tee "$LOG_DIR/rate_${RATE}.log"

    echo "[INFO] Rate=${RATE} completed. Log saved to $LOG_DIR/rate_${RATE}.log"
done

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  1단계 탐색 완료! 로그 위치: $LOG_DIR/"
echo "═══════════════════════════════════════════════════════════════"

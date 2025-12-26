#!/bin/bash
# vLLM 服务启动脚本

set -e

# 默认参数
MODEL_PATH=${MODEL_PATH:-"/app/models/llama3-law-merged"}
HOST=${HOST:-"0.0.0.0"}
PORT=${PORT:-8000}
DTYPE=${DTYPE:-"bfloat16"}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.85}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-128}

echo "🚀 启动 vLLM 服务..."
echo "📁 模型路径: $MODEL_PATH"
echo "🌐 服务地址: $HOST:$PORT"

# 检查模型是否存在
if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ 错误: 模型路径不存在: $MODEL_PATH"
    echo "   请确保模型已挂载到容器中"
    exit 1
fi

# 启动 vLLM 服务
exec vllm serve \
    "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --dtype "$DTYPE" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS"


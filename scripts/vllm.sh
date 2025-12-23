#!/bin/bash
# vLLM 服务启动脚本
# 使用方式: bash scripts/vllm.sh

# 获取脚本所在目录的父目录（项目根目录）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# 模型路径（相对于项目根目录）
MODEL_PATH="$PROJECT_ROOT/output/llama3-law-merged"

# 检查模型是否存在
if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ 错误: 模型路径不存在: $MODEL_PATH"
    echo "请先运行训练和合并脚本生成模型"
    exit 1
fi

echo "🚀 启动 vLLM 服务..."
echo "📁 模型路径: $MODEL_PATH"

vllm serve \
    "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 4096 \
    --max-num-seqs 128
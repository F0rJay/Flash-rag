#!/bin/bash
# 模型合并脚本
# 将 LoRA 权重合并到基础模型中

set -e

echo "🔄 开始合并 LoRA 权重到基础模型..."
echo ""

cd "$(dirname "$0")/.."

# 检查 LoRA 适配器是否存在
ADAPTER_PATH="output/llama3-law-assistant-lora"
if [ ! -d "$ADAPTER_PATH" ]; then
    echo "❌ 错误: LoRA 适配器不存在: $ADAPTER_PATH"
    echo "   请确保训练已完成并保存了适配器"
    exit 1
fi

echo "✅ 找到 LoRA 适配器: $ADAPTER_PATH"
echo ""

# 运行合并脚本
echo "📦 开始合并（这可能需要几分钟）..."
python src/training/merge.py

echo ""
echo "✅ 合并完成！"
echo ""
echo "📁 合并后的模型保存在: output/llama3-law-merged/"
echo ""
echo "🚀 下一步: 可以使用 vLLM 部署合并后的模型了！"


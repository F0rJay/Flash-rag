#!/bin/bash
# 训练脚本（带 OOM 修复）
# 设置 PyTorch CUDA 内存分配优化

set -e

echo "🚀 启动训练（已优化显存使用）"
echo ""

# 设置 PyTorch CUDA 内存分配优化（减少内存碎片）
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "📝 显存优化设置:"
echo "  - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
echo ""

# 检查配置
echo "📋 当前训练配置:"
echo "  - 批次大小: $(grep 'per_device_train_batch_size' config/train_config.yaml | awk '{print $2}')"
echo "  - 梯度累积: $(grep 'gradient_accumulation_steps' config/train_config.yaml | awk '{print $2}')"
echo "  - 序列长度: $(grep 'max_seq_length' config/train_config.yaml | awk '{print $2}')"
echo "  - LoRA rank: $(grep '^  r:' config/train_config.yaml | awk '{print $2}')"
echo "  - 梯度检查点: $(grep 'gradient_checkpointing' config/train_config.yaml | awk '{print $2}')"
echo ""

# 运行训练
cd "$(dirname "$0")/.."
python train.py


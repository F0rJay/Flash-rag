#!/bin/bash
# TensorBoard 可视化启动脚本

# 获取项目根目录
PROJECT_ROOT=$(dirname "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)")

# TensorBoard 日志目录
LOG_DIR="${PROJECT_ROOT}/output/logs"

# 检查日志目录是否存在
if [ ! -d "$LOG_DIR" ]; then
    echo "❌ 错误: TensorBoard 日志目录不存在: $LOG_DIR"
    echo "   请先运行训练脚本生成日志"
    exit 1
fi

echo "🚀 启动 TensorBoard..."
echo "📊 日志目录: $LOG_DIR"
echo "🌐 访问地址: http://localhost:6006"
echo ""
echo "按 Ctrl+C 停止 TensorBoard"
echo ""

# 启动 TensorBoard
tensorboard --logdir "$LOG_DIR" --port 6006 --host 0.0.0.0


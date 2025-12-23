#!/bin/bash
# FastAPI 服务启动脚本
# 使用方式: bash scripts/fastapi.sh

# 获取脚本所在目录的父目录（项目根目录）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# 切换到项目根目录
cd "$PROJECT_ROOT"

echo "🚀 启动 FastAPI 服务..."
echo "📁 项目根目录: $PROJECT_ROOT"

# 启动 FastAPI，指定模块路径
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8080
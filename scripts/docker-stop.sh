#!/bin/bash
# Docker Compose 停止脚本

set -e

# 获取脚本所在目录的父目录（项目根目录）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

echo "🛑 停止 LegalFlash-RAG 服务..."

docker-compose down

echo "✅ 服务已停止"


#!/bin/bash
# Docker Compose 启动脚本

set -e

# 获取脚本所在目录的父目录（项目根目录）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

echo "🐳 启动 LegalFlash-RAG 微服务架构..."
echo ""

# 检查 Docker 和 Docker Compose
if ! command -v docker &> /dev/null; then
    echo "❌ 错误: Docker 未安装"
    echo "   安装: curl -fsSL https://get.docker.com | sh"
    exit 1
fi

if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "❌ 错误: Docker Compose 未安装"
    echo "   安装: pip install docker-compose"
    exit 1
fi

# 检查 NVIDIA Docker（GPU 支持）
if ! docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    echo "⚠️  警告: NVIDIA Docker 可能未正确配置，GPU 支持可能不可用"
    echo "   如果遇到问题，请检查: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
fi

# 检查模型是否存在
MODEL_PATH="$PROJECT_ROOT/output/llama3-law-merged"
if [ ! -d "$MODEL_PATH" ]; then
    echo "⚠️  警告: 模型路径不存在: $MODEL_PATH"
    echo "   请确保已完成模型训练和权重合并"
    echo "   或者修改 docker-compose.yml 中的 MODEL_PATH"
    read -p "是否继续？(y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 构建镜像
echo "🔨 构建 Docker 镜像..."
docker-compose build

# 启动服务
echo ""
echo "🚀 启动服务..."
docker-compose up -d

# 显示服务状态
echo ""
echo "📊 服务状态:"
docker-compose ps

echo ""
echo "✅ 服务已启动！"
echo ""
echo "📝 服务地址:"
echo "   - vLLM API: http://localhost:8000"
echo "   - FastAPI:  http://localhost:8080"
echo "   - Streamlit: http://localhost:8501"
echo ""
echo "📋 常用命令:"
echo "   - 查看日志: docker-compose logs -f"
echo "   - 停止服务: docker-compose down"
echo "   - 重启服务: docker-compose restart"
echo "   - 查看状态: docker-compose ps"


#!/bin/bash
# App 服务启动脚本（FastAPI + Streamlit）

set -e

# 等待 vLLM 服务就绪
echo "⏳ 等待 vLLM 服务启动..."
VLLM_URL=${VLLM_URL:-"http://vllm-service:8000"}
MAX_RETRIES=30
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -s "$VLLM_URL/health" > /dev/null 2>&1; then
        echo "✅ vLLM 服务已就绪"
        break
    fi
    RETRY_COUNT=$((RETRY_COUNT + 1))
    echo "   等待中... ($RETRY_COUNT/$MAX_RETRIES)"
    sleep 2
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo "⚠️  警告: vLLM 服务未就绪，但继续启动 App 服务"
fi

# 启动 FastAPI 服务（后台）
echo "🚀 启动 FastAPI 服务..."
uvicorn src.api.main:app \
    --host 0.0.0.0 \
    --port 8080 \
    --workers 1 \
    &

# 等待 FastAPI 启动
sleep 3

# 启动 Streamlit 服务（前台）
echo "🚀 启动 Streamlit 前端..."
exec streamlit run src/frontend/frontend.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true


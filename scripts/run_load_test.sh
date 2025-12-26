#!/bin/bash
# Locust 压测脚本
# 使用方法: bash scripts/run_load_test.sh

set -e

echo "🚀 启动 LegalFlash-RAG 压测"
echo ""

# 检查 Locust 是否安装
if ! command -v locust &> /dev/null; then
    echo "❌ Locust 未安装，正在安装..."
    pip install locust
fi

# 设置默认参数
HOST="${HOST:-http://localhost:8080}"
USERS="${USERS:-10}"
SPAWN_RATE="${SPAWN_RATE:-2}"
DURATION="${DURATION:-60s}"

echo "📊 压测配置:"
echo "  - 目标地址: $HOST"
echo "  - 并发用户数: $USERS"
echo "  - 用户增长速率: $SPAWN_RATE/秒"
echo "  - 持续时间: $DURATION"
echo ""

# 检查服务是否可用
echo "🔍 检查服务状态..."
if curl -f -s "$HOST/health" > /dev/null; then
    echo "✅ 服务可用"
else
    echo "❌ 服务不可用，请先启动 FastAPI 服务"
    exit 1
fi

echo ""
echo "🌐 启动 Locust Web UI..."
echo "   访问 http://localhost:8089 进行压测"
echo "   或使用命令行模式（无头模式）"
echo ""

# 启动 Locust
cd "$(dirname "$0")/.."
locust -f tests/locustfile.py \
    --host="$HOST" \
    --users="$USERS" \
    --spawn-rate="$SPAWN_RATE" \
    --run-time="$DURATION" \
    --headless \
    --html=reports/locust_report.html \
    --csv=reports/locust_stats

echo ""
echo "✅ 压测完成！"
echo "📊 报告已保存到:"
echo "   - HTML: reports/locust_report.html"
echo "   - CSV: reports/locust_stats.csv"


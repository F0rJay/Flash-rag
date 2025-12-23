#!/bin/bash
# 检查 vLLM 服务是否在运行

echo "=== 检查 vLLM 服务状态 ==="
echo ""

# 1. 检查进程
echo "1. 检查进程："
if ps aux | grep -E "vllm serve" | grep -v grep > /dev/null; then
    echo "   ✅ vLLM 进程正在运行"
    ps aux | grep -E "vllm serve" | grep -v grep
else
    echo "   ❌ vLLM 进程未运行"
fi
echo ""

# 2. 检查端口
echo "2. 检查端口 8000："
if netstat -tlnp 2>/dev/null | grep 8000 > /dev/null || ss -tlnp 2>/dev/null | grep 8000 > /dev/null; then
    echo "   ✅ 端口 8000 已被占用"
    netstat -tlnp 2>/dev/null | grep 8000 || ss -tlnp 2>/dev/null | grep 8000
else
    echo "   ❌ 端口 8000 未被占用"
fi
echo ""

# 3. 测试 API 连接
echo "3. 测试 API 连接："
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "   ✅ API 健康检查通过"
    curl -s http://localhost:8000/health | head -3
elif curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
    echo "   ✅ API 可以访问（模型列表）"
    curl -s http://localhost:8000/v1/models | head -5
else
    echo "   ❌ 无法连接到 vLLM API"
    echo "   提示：请运行 'bash vllm.sh' 启动服务"
fi
echo ""

echo "=== 检查完成 ==="


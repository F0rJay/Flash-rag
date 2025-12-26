# vLLM 服务 Dockerfile
# 专门负责模型推理计算

FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# 设置工作目录
WORKDIR /app

# 安装 Python 和基础依赖
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# 设置 Python 链接
RUN ln -s /usr/bin/python3 /usr/bin/python

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV HF_ENDPOINT=https://hf-mirror.com

# 复制依赖文件
COPY requirements.txt /app/

# 安装 Python 依赖（优先安装 vLLM 相关）
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple \
    torch>=2.0.0 \
    transformers>=4.30.0 \
    vllm>=0.3.0 \
    && pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple \
    -r requirements.txt

# 复制启动脚本
COPY docker/vllm/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# 暴露端口
EXPOSE 8000

# 启动命令
ENTRYPOINT ["/app/entrypoint.sh"]


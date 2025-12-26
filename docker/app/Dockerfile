# App 服务 Dockerfile
# 包含 FastAPI (RAG逻辑) 和 Streamlit (前端)

FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV HF_ENDPOINT=https://hf-mirror.com
ENV VLLM_URL=http://vllm-service:8000

# 复制依赖文件
COPY requirements.txt /app/

# 安装 Python 依赖
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple \
    -r requirements.txt

# 复制项目代码
COPY . /app/

# 创建必要的目录
RUN mkdir -p /app/data/datasets /app/data/docs /app/output /app/logs

# 复制启动脚本
COPY docker/app/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# 暴露端口
# 8080: FastAPI
# 8501: Streamlit
EXPOSE 8080 8501

# 启动命令
ENTRYPOINT ["/app/entrypoint.sh"]


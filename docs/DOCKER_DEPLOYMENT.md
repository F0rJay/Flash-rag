# Docker 部署指南

## 📋 前置要求

### 1. 系统要求

- **操作系统**: Linux (Ubuntu 20.04+ 推荐)
- **GPU**: NVIDIA GPU (支持 CUDA 12.1+)
- **内存**: 至少 32GB RAM
- **存储**: 至少 100GB 可用空间（用于模型和向量数据库）

### 2. 软件安装

#### 安装 Docker

```bash
# 安装 Docker
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER

# 重新登录或执行
newgrp docker

# 验证安装
docker --version
```

#### 安装 Docker Compose

```bash
# 方法1: 使用 pip 安装
pip install docker-compose

# 方法2: 使用官方脚本
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# 验证安装
docker-compose --version
```

#### 安装 NVIDIA Docker（GPU 支持）

```bash
# 添加 NVIDIA Docker 仓库
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# 安装 NVIDIA Docker
sudo apt-get update
sudo apt-get install -y nvidia-docker2

# 重启 Docker
sudo systemctl restart docker

# 验证 GPU 支持
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

## 🚀 快速开始

### 1. 准备模型

确保已完成模型训练和权重合并：

```bash
# 检查模型是否存在
ls -lh output/llama3-law-merged/
```

如果模型不存在，请先完成：
1. 模型训练：`python src/training/train.py`
2. 权重合并：`python src/training/merge.py`

### 2. 准备知识库（可选）

如果已有向量数据库，确保在以下目录：
- `chroma_db/` - 法条型知识库
- `chroma_db_case/` - 案例型知识库
- `chroma_db_judgement/` - 判决书型知识库

如果没有，可以稍后构建。

### 3. 启动服务

```bash
# 使用启动脚本（推荐）
bash scripts/docker-start.sh

# 或手动启动
docker-compose up -d
```

### 4. 验证服务

```bash
# 检查服务状态
docker-compose ps

# 检查 vLLM 服务
curl http://localhost:8000/health

# 检查 FastAPI 服务
curl http://localhost:8080/health

# 访问前端
# 浏览器打开: http://localhost:8501
```

## 📊 服务架构

```
┌─────────────────────────────────────────┐
│         Docker Compose Network          │
│                                         │
│  ┌──────────────┐    ┌──────────────┐  │
│  │ vllm-service │    │ app-service  │  │
│  │              │    │              │  │
│  │ Port: 8000   │◄───┤ FastAPI:8080 │  │
│  │ GPU: 1x      │    │ Streamlit:   │  │
│  │              │    │   8501       │  │
│  └──────────────┘    └──────────────┘  │
│                                         │
└─────────────────────────────────────────┘
```

## ⚙️ 配置说明

### 环境变量

**vLLM 服务：**
- `MODEL_PATH`: 模型路径（默认: `/app/models/llama3-law-merged`）
- `HOST`: 服务地址（默认: `0.0.0.0`）
- `PORT`: 服务端口（默认: `8000`）
- `DTYPE`: 数据类型（默认: `bfloat16`）
- `GPU_MEMORY_UTILIZATION`: 显存使用率（默认: `0.85`）
- `MAX_MODEL_LEN`: 最大序列长度（默认: `4096`）
- `MAX_NUM_SEQS`: 最大并发序列数（默认: `128`）

**App 服务：**
- `VLLM_URL`: vLLM 服务地址（默认: `http://vllm-service:8000`）
- `HF_ENDPOINT`: HuggingFace 镜像（默认: `https://hf-mirror.com`）

### 自定义配置

复制 `docker-compose.override.yml.example` 为 `docker-compose.override.yml`：

```bash
cp docker-compose.override.yml.example docker-compose.override.yml
```

然后编辑 `docker-compose.override.yml` 来自定义配置。

## 🔧 常用操作

### 查看日志

```bash
# 查看所有服务日志
docker-compose logs -f

# 查看特定服务日志
docker-compose logs -f vllm-service
docker-compose logs -f app-service
```

### 重启服务

```bash
# 重启所有服务
docker-compose restart

# 重启特定服务
docker-compose restart vllm-service
docker-compose restart app-service
```

### 停止服务

```bash
# 停止服务（保留容器）
docker-compose stop

# 停止并删除容器
docker-compose down

# 停止并删除容器和卷
docker-compose down -v
```

### 更新服务

```bash
# 重新构建镜像
docker-compose build

# 重新构建并启动
docker-compose up -d --build
```

## 🐛 故障排查

### 问题 1: GPU 不可用

**症状：** vLLM 服务启动失败，提示 GPU 相关错误

**解决：**
```bash
# 检查 NVIDIA Docker 是否正确安装
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# 如果失败，重新安装 NVIDIA Docker
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### 问题 2: 模型路径不存在

**症状：** vLLM 服务启动失败，提示模型路径不存在

**解决：**
1. 检查模型是否已训练和合并
2. 修改 `docker-compose.yml` 中的 `MODEL_PATH` 环境变量
3. 确保 volume 挂载路径正确

### 问题 3: 端口冲突

**症状：** 服务启动失败，提示端口已被占用

**解决：**
```bash
# 检查端口占用
sudo lsof -i :8000
sudo lsof -i :8080
sudo lsof -i :8501

# 修改 docker-compose.yml 中的端口映射
ports:
  - "8001:8000"  # 改为其他端口
```

### 问题 4: 容器无法连接

**症状：** App 服务无法连接到 vLLM 服务

**解决：**
1. 检查服务是否在同一网络：`docker network ls`
2. 检查 vLLM 服务健康状态：`curl http://localhost:8000/health`
3. 检查环境变量 `VLLM_URL` 是否正确

## 📈 性能优化

### GPU 资源优化

```yaml
# docker-compose.override.yml
services:
  vllm-service:
    environment:
      - GPU_MEMORY_UTILIZATION=0.9  # 提高显存使用率
      - MAX_NUM_SEQS=256  # 增加并发数
```

### 内存优化

```yaml
services:
  app-service:
    deploy:
      resources:
        limits:
          memory: 8G
        reservations:
          memory: 4G
```

## 🔒 生产环境建议

1. **使用环境变量文件**: 创建 `.env` 文件存储敏感配置
2. **启用 HTTPS**: 使用 Nginx 反向代理，配置 SSL 证书
3. **日志管理**: 配置日志轮转和集中日志管理
4. **监控告警**: 集成 Prometheus + Grafana 监控
5. **备份策略**: 定期备份模型和向量数据库
6. **资源限制**: 设置合理的 CPU 和内存限制

## 📚 参考资源

- [Docker 官方文档](https://docs.docker.com/)
- [Docker Compose 文档](https://docs.docker.com/compose/)
- [NVIDIA Docker 文档](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
- [vLLM 文档](https://docs.vllm.ai/)


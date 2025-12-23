## Flash-RAG

Flash-RAG 是一个基于 **vLLM** 的高并发垂直领域智能问答引擎，当前主要聚焦于 **法律条文咨询助手** 场景。

---

## 项目结构

```text
Flash-RAG/
├── src/                   # 源代码目录
│   ├── core/              # 核心功能模块
│   │   ├── CustomVLLM.py  # 自定义 vLLM 集成
│   │   └── ingest.py      # 文档向量化处理
│   ├── api/               # API 服务
│   │   └── main.py       # FastAPI RAG 服务
│   ├── training/          # 训练相关
│   │   ├── train.py      # 模型训练脚本
│   │   ├── evaluate.py   # 模型评估脚本
│   │   └── merge.py      # 权重合并脚本
│   └── frontend/         # 前端相关
│       └── frontend.py
├── scripts/              # 脚本目录
│   ├── vllm.sh           # vLLM 服务启动脚本
│   ├── fastapi.sh        # FastAPI 服务启动脚本
│   ├── check_vllm.sh     # vLLM 服务检查脚本
│   ├── frontend.sh       # 前端启动脚本
│   ├── view_training.sh  # TensorBoard 可视化启动脚本
│   ├── prepare_dataset.py        # 数据集准备脚本（转换和划分）
│   ├── prepare_rag_knowledge.py   # RAG 知识库准备脚本（提取法律条文）
│   └── analyze_dataset.py        # 数据集分析和验证脚本
├── config/               # 配置文件目录
│   └── train_config.yaml # 训练与模型相关的全部参数
├── data/                 # 数据目录
│   ├── datasets/         # 训练/评测数据
│   │   ├── train.jsonl   # 训练集
│   │   ├── val.jsonl     # 验证集
│   │   └── test.jsonl    # 测试集
│   └── docs/             # 文档数据（RAG 知识库）
│       ├── legal_docs.txt      # 法条型知识库（法律条文）
│       ├── case_docs.txt       # 案例型知识库（案件+判决）
│       └── judgement_docs.txt  # 判决书型知识库（完整判决书）
├── tests/                # 测试文件
│   └── test_client.py    # API 测试客户端
├── output/               # 训练输出与日志（自动生成，已加入 .gitignore）
├── chroma_db/            # 法条型向量数据库（自动生成，已加入 .gitignore）
├── chroma_db_case/       # 案例型向量数据库（自动生成，已加入 .gitignore）
├── chroma_db_judgement/  # 判决书型向量数据库（自动生成，已加入 .gitignore）
├── requirements.txt      # 项目依赖
├── .gitignore           # Git 忽略规则
└── README.md            # 项目说明文档
```

---

## 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone https://github.com/F0rJay/Flash-rag.git
cd Flash-rag

# 安装依赖
pip install -r requirements.txt
```

### 2. 准备数据

#### 2.1 训练数据准备

项目需要的数据格式为 JSONL，每行一个 JSON 对象，包含以下字段：
```json
{
  "instruction": "问题或指令",
  "input": "上下文或输入（可为空字符串）",
  "output": "期望的回答"
}
```

**如果你有 DISC-Law 格式的数据**（格式：`{"id": "...", "input": "...", "output": "..."}`），可以使用项目提供的脚本自动转换和划分：

```bash
# 转换 DISC-Law 格式并划分数据集
python scripts/prepare_dataset.py /path/to/DISC-Law-SFT-Pair-QA-released.jsonl \
    --train-ratio 0.8 \
    --val-ratio 0.1 \
    --test-ratio 0.1
```

脚本会自动：
- 将 DISC-Law 格式转换为项目格式（`input` → `instruction`）
- 按比例划分数据集（默认：训练集 80%，验证集 10%，测试集 10%）
- 保存到 `data/datasets/` 目录：
  - `train.jsonl` - 训练集
  - `val.jsonl` - 验证集
  - `test.jsonl` - 测试集

**如果你已有符合格式的数据**，直接放到 `data/datasets/` 目录即可：
- `train.jsonl` - 训练集
- `val.jsonl` - 验证集（可选，用于训练过程中的评估）
- `test.jsonl` - 测试集（可选，用于最终评估）

**验证已有数据集格式：**
```bash
# 验证数据集格式是否正确
python scripts/prepare_dataset.py --validate

# 或使用分析脚本获取详细统计
python scripts/analyze_dataset.py
```

**使用已有数据集（不进行转换）：**
```bash
# 如果已有 train/val/test.jsonl，直接使用
python scripts/prepare_dataset.py --use-existing
```

#### 2.2 RAG 知识库文档

**方法 1: 使用 DISC-Law 数据集（推荐）**

项目支持从 DISC-Law JSONL 文件构建两种类型的知识库：

**法条型知识库（法律条文）：**
```bash
# 提取法律条文（从 reference 字段）
python scripts/prepare_rag_knowledge.py \
    /path/to/DISC-Law-SFT-Triplet-QA-released.jsonl \
    --mode law \
    --output data/docs/legal_docs.txt
```

**案例型知识库（案件+判决）：**
```bash
# 提取案例（从 input + output 字段）
python scripts/prepare_rag_knowledge.py \
    /path/to/DISC-Law-SFT-Triplet-released.jsonl \
    --mode case \
    --output data/docs/case_docs.txt
```

**判决书型知识库（完整判决书）：**
```bash
# 提取判决书（从 input 字段，包含完整判决书原文）
python scripts/prepare_rag_knowledge.py \
    /path/to/DISC-Law-SFT-Pair.jsonl \
    --mode judgement \
    --output data/docs/judgement_docs.txt
```

**混合模式（同时提取法条和案例）：**
```bash
python scripts/prepare_rag_knowledge.py \
    file1.jsonl file2.jsonl \
    --mode mixed \
    --output data/docs/mixed_docs.txt
```

脚本功能：
- `--mode law`: 提取 `reference` 字段中的法律条文
- `--mode case`: 提取 `input`（案件事实）+ `output`（判决结果）
- `--mode judgement`: 提取 `input`（完整判决书原文，包含案件事实、判决结果、法律条文等）
- `--mode mixed`: 同时提取法条和案例
- 自动去重并合并多个文件

**方法 2: 手动准备**

直接准备文本文件：
- `data/docs/legal_docs.txt` - 法律条文（每行或每段一个条文）
- `data/docs/case_docs.txt` - 案例文档（案件事实+判决结果）
- `data/docs/judgement_docs.txt` - 判决书文档（完整判决书原文）

### 3. 模型训练与部署

#### 步骤 1: 训练 LoRA 适配器

```bash
# 从项目根目录运行
python src/training/train.py
```

训练配置在 `config/train_config.yaml` 中，可根据需要调整：
- 模型路径
- 数据路径（训练集、验证集、测试集）
- 训练参数（学习率、批次大小、训练轮数）
- 评估设置（评估频率、保存最佳模型等）
- LoRA 参数（rank、alpha 等）

**GPU 监控：**

训练过程中会自动监控 GPU 状态：
- 💾 显存使用（已分配/预留/总显存）
- ⚡ GPU 使用率
- 🌡️  温度监控
- 🔋 功耗监控

监控数据会：
- 定期打印到控制台（每 10 步）
- 实时记录到 TensorBoard

**训练可视化（TensorBoard）：**

训练过程中会自动记录训练指标到 TensorBoard：
```bash
# 启动 TensorBoard（在另一个终端）
bash scripts/view_training.sh

# 或手动启动
tensorboard --logdir output/logs --port 6006
```

然后在浏览器中访问 `http://localhost:6006` 查看：
- 📈 训练损失曲线
- 📊 验证损失曲线
- 📉 学习率变化
- ⏱️  训练速度（samples/sec）
- 🖥️  GPU 指标（显存、使用率、温度、功耗）

**验证集评估：**

训练脚本会自动使用验证集进行评估（如果配置了 `val_path`）：
- 每 `eval_steps` 步评估一次
- 自动保存最佳模型（基于 `eval_loss`）
- 训练日志中包含验证集指标
- 训练统计信息保存在 `output/training_stats.json`

**查看训练日志：**
```bash
# 训练日志保存在 output/ 目录
ls output/

# 查看训练统计
cat output/training_stats.json
```

#### 步骤 2: 合并权重（必须！）

```bash
python src/training/merge.py
```

合并后的模型将保存在 `output/llama3-law-merged/` 目录。

#### 步骤 3: 文档向量化（RAG 知识库构建）

项目支持两种类型的 RAG 知识库：

**3.1 法条型知识库（法律条文）**

```bash
# 从 DISC-Law JSONL 文件提取法律条文
python scripts/prepare_rag_knowledge.py \
    /path/to/DISC-Law-SFT-Triplet-QA-released.jsonl \
    --mode law \
    --output data/docs/legal_docs.txt

# 构建法条型向量数据库
python src/core/ingest.py \
    --docs-path data/docs/legal_docs.txt \
    --knowledge-type law \
    --chunk-size 500 \
    --chunk-overlap 50
```

**3.2 案例型知识库（案件+判决）**

```bash
# 从 DISC-Law JSONL 文件提取案例
python scripts/prepare_rag_knowledge.py \
    /path/to/DISC-Law-SFT-Triplet-released.jsonl \
    --mode case \
    --output data/docs/case_docs.txt

# 构建案例型向量数据库
python src/core/ingest.py \
    --docs-path data/docs/case_docs.txt \
    --knowledge-type case \
    --chunk-size 1000 \
    --chunk-overlap 100
```

**3.3 判决书型知识库（完整判决书）**

```bash
# 从 DISC-Law JSONL 文件提取判决书
python scripts/prepare_rag_knowledge.py \
    /path/to/DISC-Law-SFT-Pair.jsonl \
    --mode judgement \
    --output data/docs/judgement_docs.txt

# 构建判决书型向量数据库（使用更大的 chunk_size 保持判决书完整性）
python src/core/ingest.py \
    --docs-path data/docs/judgement_docs.txt \
    --knowledge-type judgement \
    --chunk-size 1500 \
    --chunk-overlap 150
```

**3.4 混合模式（推荐）**

同时构建多种知识库，API 会自动启用混合检索：
- 法条型：提供法律依据
- 案例型：提供相似案例参考
- 判决书型：提供完整判决书参考

```bash
# 1. 准备法条型知识库
python scripts/prepare_rag_knowledge.py file1.jsonl --mode law
python src/core/ingest.py --knowledge-type law

# 2. 准备案例型知识库
python scripts/prepare_rag_knowledge.py file2.jsonl --mode case
python src/core/ingest.py --knowledge-type case

# 3. 准备判决书型知识库
python scripts/prepare_rag_knowledge.py file3.jsonl --mode judgement
python src/core/ingest.py --knowledge-type judgement

# 4. 启动服务（自动启用混合检索）
bash scripts/fastapi.sh
```

**知识库说明：**
- 法条型：存储位置 `chroma_db/`，包含法律条文原文
- 案例型：存储位置 `chroma_db_case/`，包含案件事实和判决结果
- 判决书型：存储位置 `chroma_db_judgement/`，包含完整判决书（案件事实+判决结果+法律条文）
- 混合检索：同时从多个知识库检索，结合法条、案例和判决书给出更全面的回答

**验证集评估：**

训练脚本会自动使用验证集进行评估（如果配置了 `val_path`）：
- 每 `eval_steps` 步评估一次
- 自动保存最佳模型（基于 `eval_loss`）
- 训练日志中包含验证集指标

**查看训练日志：**
```bash
# 训练日志保存在 output/ 目录
ls output/
```

### 5. 启动服务

#### 启动 vLLM 推理服务（终端 1）

```bash
bash scripts/vllm.sh
```

服务将在 `http://localhost:8000` 启动。

**检查服务状态：**
```bash
bash scripts/check_vllm.sh
```

#### 启动 FastAPI RAG 服务（终端 2）

```bash
bash scripts/fastapi.sh
```

服务将在 `http://localhost:8080` 启动。

### 6. 测试 API

```bash
# 使用测试客户端
python tests/test_client.py

# 或使用 curl
curl -X POST http://localhost:8080/api/rag/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "如果甲方逾期支付本金，需要承担什么违约责任？"}'
```

### 配置说明

所有配置都在 `config/train_config.yaml` 中，包括：
- 模型配置（模型名称、最大序列长度）
- 数据配置（训练数据路径）
- 训练参数（学习率、批次大小、训练轮数）
- LoRA 参数（rank、alpha、dropout）
- 量化配置（是否启用 4-bit 量化）

---

## 🚀 项目开发要点速查卡

### 核心目标

打造一个 **低延迟、高并发、懂垂直领域知识** 的生产级 AI 问答系统。

---

### Phase 1: 模型特训 (Training & Optimization)

**任务：** 让模型"懂行"且"轻量"。

**技术栈：** HuggingFace Transformers, PEFT, AutoGPTQ / BitsAndBytes

#### 关键概念

| 概念 | 说明 | 关键参数 |
|------|------|----------|
| **LoRA (Low-Rank Adaptation)** | 只训练旁路小矩阵，大幅减少训练成本 | `r` (Rank, 如 8 或 16)<br>`target_modules` (通常涵盖所有 Linear layers) |
| **Merge Weights (权重合并)** | ⚠️ **必做步骤！** 训练完必须将 LoRA 权重合并回底座模型 | 保存为独立的 `.safetensors` 格式 |
| **Quantization (量化)** | 推荐 AWQ 格式（比 GPTQ 对 vLLM 支持更好） | 将显存需求砍到 1/3 |

#### ⚠️ 避坑指南

> **重要：** 只有合并了权重，推理速度才会快。挂载 Adapter 推理反而会变慢。

**训练流程：**
```bash
# 1. 训练 LoRA 适配器
python src/training/train.py

# 2. 合并权重（必须！）
python src/training/merge.py

# 3. 量化（可选，但推荐）
# 使用 AutoGPTQ 或 AWQ 工具进行量化
# 量化后的模型路径需要在 vllm.sh 中指定
```

---

### Phase 2: 极速推理 (Inference Engine)

**任务：** 榨干 GPU 性能，解决显存瓶颈。

**技术栈：** vLLM

#### 核心机制

- **PagedAttention**: 显存分页管理，拒绝碎片化

#### 启动参数示例

使用项目提供的脚本（推荐）：
```bash
bash scripts/vllm.sh
```

脚本会自动：
- 检测模型路径（`output/llama3-law-merged`）
- 设置合适的显存使用率（0.85）
- 配置并发限制（max-num-seqs 128）

手动启动（如需自定义参数）：
```bash
vllm serve \
    output/llama3-law-merged \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype bfloat16 \
    --quantization awq \          # 如果模型量化过，必须加
    --gpu-memory-utilization 0.85 \ # 显存预留比例，越大 KV Cache 越多
    --max-model-len 4096 \        # 强制截断，防止 OOM
    --max-num-seqs 128            # 限制并发序列数
```

#### 性能调优

| 指标 | 说明 | 平衡策略 |
|------|------|----------|
| **Throughput (吞吐量)** | 单位时间处理的请求数 | Batch size 越大，吞吐越高 |
| **Latency (延迟)** | 单个请求的响应时间 | 但延迟可能增加，需寻找平衡点 |

#### ⚠️ 避坑指南

> **常见错误：** 遇到 `Request ignored` 报错，通常是：
> - `max-model-len` 没设限制
> - 显存被 KV Cache 撑爆了
> - 需要降低 `gpu-memory-utilization` 或 `max-num-seqs`

---

### Phase 3: 后端架构 (Backend & RAG)

**任务：** 搭建不阻塞的 API，实现打字机效果。

**技术栈：** FastAPI, Uvicorn, LangChain / LlamaIndex

#### 核心模式

- **Async/Await**: 必须使用 `async def` 定义接口，调用数据库和模型时必须 `await`
- **SSE (Server-Sent Events)**: 流式输出的标准协议

#### RAG 黄金链路

```mermaid
graph LR
    A[用户问题] --> B[Rewrite<br/>改写问题]
    B --> C[Retrieve<br/>混合检索]
    C --> D[Rerank<br/>重排序]
    D --> E[Generate<br/>生成答案]
```

1. **Rewrite**: 改写用户问题，提升检索准确率
2. **Retrieve**: 混合检索（Vector + Keyword）
3. **Rerank (重排序)**: 使用 BGE-Reranker 等小模型对检索结果精排（Top 50 → Top 5）
4. **Generate**: 拼接 Prompt 送入 vLLM

**当前实现：**

项目已实现基础的 RAG 流程（位于 `src/api/main.py`）：
- ✅ 向量检索（使用 ChromaDB）
- ✅ 上下文拼接
- ✅ vLLM 集成

**扩展方向：**
```python
# 在 src/api/main.py 中扩展
@app.post("/api/rag/chat")
async def chat_endpoint(request: ChatRequest):
    # 1. 改写问题（待实现）
    rewritten_query = await rewrite_query(request.query)
    
    # 2. 检索（已实现）
    docs = await retriever.retrieve(rewritten_query)
    
    # 3. 重排序（待实现）
    ranked_docs = await reranker.rerank(docs, top_k=5)
    
    # 4. 生成（已实现）
    response = await llm.generate(context=ranked_docs, query=request.query)
    
    return {"response": response}
```

---

### Phase 4: 生产交付 (Production & Ops)

**任务：** 证明系统稳健，用数据说话。

**技术栈：** Docker, Locust (压测), Prometheus + Grafana

#### 监控重点

| 指标 | 说明 | 阈值 |
|------|------|------|
| **gpu_cache_usage** | KV Cache 使用率 | 如果长期高于 95%，说明需要加卡或优化模型长度 |
| **request_latency** | 请求延迟 | P50 < 200ms, P99 < 1s |
| **throughput** | 吞吐量 | 根据业务需求设定 |

#### 部署检查清单

- [ ] 模型权重已合并（非 LoRA Adapter）
- [ ] vLLM 服务正常启动，无 OOM 错误
- [ ] FastAPI 接口支持异步和流式输出
- [ ] RAG 链路完整（Rewrite → Retrieve → Rerank → Generate）
- [ ] 监控指标已配置（GPU 使用率、延迟、吞吐量）
- [ ] 压测通过（使用 Locust 进行负载测试）

---

## 📚 相关资源

- [vLLM 官方文档](https://docs.vllm.ai/)
- [LangChain 文档](https://python.langchain.com/)
- [PEFT (LoRA) 文档](https://huggingface.co/docs/peft/)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/)

## 🔧 常见问题

### Q: 训练时出现显存不足？
A: 在 `config/train_config.yaml` 中：
- 启用 4-bit 量化：`load_in_4bit: true`
- 减小批次大小：`per_device_train_batch_size: 4`
- 增加梯度累积：`gradient_accumulation_steps: 2`

### Q: vLLM 启动失败，提示 OOM？
A: 在 `scripts/vllm.sh` 中：
- 降低 `--gpu-memory-utilization`（如 0.8）
- 减小 `--max-num-seqs`（如 64）
- 减小 `--max-model-len`（如 2048）

### Q: DISC-Law 数据集格式能直接用吗？
A: 不能直接使用。DISC-Law 格式是 `{"id": "...", "input": "...", "output": "..."}`，而项目需要 `{"instruction": "...", "input": "...", "output": "..."}` 格式。

**解决方法：**
```bash
# 方法1: 转换并划分数据集
python scripts/prepare_dataset.py /path/to/DISC-Law-SFT-Pair-QA-released.jsonl

# 方法2: 如果已有符合格式的数据集，直接使用
python scripts/prepare_dataset.py --use-existing

# 方法3: 验证数据集格式
python scripts/prepare_dataset.py --validate
```

脚本会自动转换格式并划分数据集。

### Q: 如何分析数据集质量？
A: 使用数据集分析脚本：

```bash
# 分析所有数据集（train/val/test）
python scripts/analyze_dataset.py

# 生成详细报告（JSON 格式）
python scripts/analyze_dataset.py --output reports/dataset_report.json
```

分析脚本会提供：
- ✅ 数据格式验证（必需字段、类型检查）
- 📊 统计信息（数量、长度分布、中位数、平均值）
- 🔍 数据质量检查（空值、重复）
- 📈 数据集报告（JSON 格式）

### Q: 如何从 DISC-Law JSONL 文件构建 RAG 知识库？
A: 项目支持三种知识库类型：

**法条型知识库（法律条文）：**
```bash
# 提取法律条文
python scripts/prepare_rag_knowledge.py file.jsonl --mode law
# 构建向量库
python src/core/ingest.py --knowledge-type law
```

**案例型知识库（案件+判决）：**
```bash
# 提取案例
python scripts/prepare_rag_knowledge.py file.jsonl --mode case
# 构建向量库
python src/core/ingest.py --knowledge-type case
```

**判决书型知识库（完整判决书）：**
```bash
# 提取判决书（从 DISC-Law-SFT-Pair.jsonl）
python scripts/prepare_rag_knowledge.py file.jsonl --mode judgement
# 构建向量库（使用更大的 chunk_size）
python src/core/ingest.py --knowledge-type judgement --chunk-size 1500 --chunk-overlap 150
```

**混合模式（推荐）：**
同时构建多种知识库，API 会自动启用混合检索，结合法条、案例和判决书给出更准确的回答。

### Q: 如何查看训练过程的可视化？
A: 使用 TensorBoard：

```bash
# 方法1: 使用脚本启动
bash scripts/view_training.sh

# 方法2: 手动启动
tensorboard --logdir output/logs --port 6006
```

然后在浏览器访问 `http://localhost:6006` 查看训练曲线（损失、学习率等）和 GPU 指标。

### Q: 如何监控 GPU 状态？
A: GPU 监控已自动启用，会：

1. **控制台输出**: 每 10 步（可配置）打印一次 GPU 状态
2. **TensorBoard**: 所有 GPU 指标实时记录，可在 `gpu/` 分组下查看

监控指标包括：
- 显存使用（已分配/预留/总显存）
- GPU 使用率
- 显存使用率
- 温度（需要安装 `nvidia-ml-py3`）
- 功耗（需要安装 `nvidia-ml-py3`）

**安装完整监控：**
```bash
pip install nvidia-ml-py3
```

**配置监控间隔：**
在 `config/train_config.yaml` 中修改 `gpu_monitor.log_interval`

### Q: 如何评估模型性能？
A: 使用评估脚本：

```bash
# 评估 LoRA 适配器
python src/training/evaluate.py --model_path output/llama3-law-assistant-lora

# 评估合并后的完整模型
python src/training/evaluate.py --model_path output/llama3-law-merged

# 快速评估（限制样本数）
python src/training/evaluate.py --model_path output/llama3-law-assistant-lora --max_samples 100
```

评估脚本会计算 BLEU、ROUGE、困惑度等指标，并生成评估报告。

### Q: 如何添加新的文档到知识库？
A: 
1. 将文档添加到 `data/docs/legal_docs.txt`（追加或替换）
2. 运行 `python src/core/ingest.py` 重新构建向量库
3. 注意：重新构建会覆盖之前的向量库

### Q: 如何修改 API 端口？
A: 
- vLLM 服务：修改 `scripts/vllm.sh` 中的 `--port`
- FastAPI 服务：修改 `scripts/fastapi.sh` 中的 `--port`

## 📝 开发说明

### 代码结构说明

- `src/core/` - 核心功能模块，可独立使用
- `src/api/` - API 服务层，依赖 core 模块
- `src/training/` - 训练相关脚本，可独立运行
- `scripts/` - 启动脚本，支持相对路径，可在任意位置运行

### 扩展开发

1. **添加新的检索器**：在 `src/core/` 中创建新模块
2. **扩展 API 接口**：在 `src/api/main.py` 中添加路由
3. **自定义训练流程**：修改 `src/training/train.py`

---

**License**: 见 [LICENSE](LICENSE) 文件


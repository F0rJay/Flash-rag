# 训练与评估指南

## 📊 训练可视化

### GPU 监控

训练过程中会自动监控 GPU 状态，包括：

- **显存使用**: 已分配/预留/总显存（MB 和百分比）
- **GPU 使用率**: GPU 计算单元使用率
- **显存使用率**: 显存带宽使用率
- **温度**: GPU 温度（摄氏度）
- **功耗**: 当前功耗和功耗限制（瓦特）

**监控方式：**

1. **控制台输出**: 每 `log_interval` 步（默认 10 步）打印一次 GPU 状态
2. **TensorBoard**: 所有 GPU 指标实时记录到 TensorBoard，可在可视化界面查看

**配置 GPU 监控：**

在 `config/train_config.yaml` 中配置：

```yaml
training:
  gpu_monitor:
    enabled: true              # 是否启用 GPU 监控
    log_interval: 10           # 打印间隔（步数）
    enable_tensorboard: true   # 是否记录到 TensorBoard
```

**安装 GPU 监控依赖（推荐）：**

```bash
# 完整 GPU 监控（包括温度、功耗等）
pip install nvidia-ml-py3

# 如果不安装，将使用基础监控（仅显存）
```

### 启动 TensorBoard

训练过程中会自动记录指标到 TensorBoard。在训练开始后，在另一个终端运行：

```bash
# 方法1: 使用脚本（推荐）
bash scripts/view_training.sh

# 方法2: 手动启动
tensorboard --logdir output/logs --port 6006
```

然后在浏览器访问 `http://localhost:6006` 查看：

- **训练损失曲线**: 监控模型学习进度
- **验证损失曲线**: 检测过拟合
- **学习率变化**: 验证学习率调度
- **训练速度**: samples/sec，监控训练效率
- **GPU 指标**: 
  - 显存使用率（每个 GPU）
  - GPU 使用率（每个 GPU）
  - 温度（每个 GPU）
  - 功耗（每个 GPU）

在 TensorBoard 中，GPU 指标位于 `gpu/` 分组下，可以按 GPU ID 查看各个 GPU 的详细指标。

### 训练统计信息

训练完成后，统计信息会自动保存到 `output/training_stats.json`：

```json
{
  "training_start": "2024-12-24T10:00:00",
  "training_end": "2024-12-24T12:30:00",
  "training_duration_hours": 2.5,
  "total_steps": 1000,
  "total_epochs": 3.0,
  "best_metric": 2.345,
  "best_model_checkpoint": "output/checkpoint-500"
}
```

## 📈 模型评估

### 基本评估

训练完成后，在测试集上评估模型性能：

```bash
# 评估 LoRA 适配器
python src/training/evaluate.py \
    --model_path output/llama3-law-assistant-lora

# 评估合并后的完整模型
python src/training/evaluate.py \
    --model_path output/llama3-law-merged
```

### 评估指标说明

评估脚本会计算以下指标：

1. **困惑度 (Perplexity)**
   - 衡量模型对测试数据的预测不确定性
   - 值越低越好，表示模型更确定
   - 计算公式: `exp(cross_entropy_loss)`

2. **BLEU 分数**
   - 评估生成文本与参考答案的 n-gram 重叠度
   - 范围: 0-1，越高越好
   - BLEU-1: 1-gram 重叠度

3. **ROUGE 分数**
   - 评估生成文本的召回率
   - ROUGE-1: 1-gram 召回率
   - ROUGE-2: 2-gram 召回率
   - ROUGE-L: 最长公共子序列
   - 范围: 0-1，越高越好

4. **平均长度**
   - 比较生成答案与参考答案的长度
   - 用于检测模型是否生成过短或过长的回答

### 快速评估

如果测试集很大，可以使用 `--max_samples` 限制评估样本数：

```bash
# 只评估前 100 个样本（用于快速测试）
python src/training/evaluate.py \
    --model_path output/llama3-law-assistant-lora \
    --max_samples 100
```

### 评估结果

评估结果会保存到 `{model_path}/evaluation_results.json`，包含：

- 所有评估指标
- 示例预测（前 5 个）
- 评估时间戳

示例输出：

```json
{
  "evaluation_time": "2024-12-24T12:30:00",
  "num_samples": 1000,
  "perplexity": 15.23,
  "bleu": {
    "bleu_1": 0.45,
    "bleu_avg": 0.45
  },
  "rouge": {
    "rouge1": 0.52,
    "rouge2": 0.38,
    "rougeL": 0.49
  },
  "average_lengths": {
    "reference": 245.6,
    "prediction": 238.2
  },
  "sample_predictions": [...]
}
```

## 🔄 完整训练流程

### 1. 准备数据

```bash
# 转换和划分数据集
python scripts/prepare_dataset.py \
    --input DISC-Law-SFT-Pair-QA-released.jsonl \
    --output-dir data/datasets

# 分析数据集
python scripts/analyze_dataset.py
```

### 2. 开始训练

```bash
# 启动训练（会自动记录到 TensorBoard）
python src/training/train.py

# 在另一个终端启动 TensorBoard
bash scripts/view_training.sh
```

### 3. 评估模型

```bash
# 评估 LoRA 适配器
python src/training/evaluate.py \
    --model_path output/llama3-law-assistant-lora

# 如果结果满意，合并权重
python src/training/merge.py

# 评估合并后的模型
python src/training/evaluate.py \
    --model_path output/llama3-law-merged
```

## 📊 性能优化建议

### 训练监控

1. **损失曲线**
   - 训练损失应持续下降
   - 验证损失应与训练损失同步下降
   - 如果验证损失开始上升，可能出现过拟合

2. **学习率**
   - 确保学习率在合理范围内（通常 1e-5 到 1e-3）
   - 如果损失不下降，尝试降低学习率

3. **训练速度**
   - 监控 samples/sec
   - 如果速度过慢，检查：
     - 批次大小是否合适
     - 梯度累积步数
     - 数据加载是否高效

### 评估指标解读

1. **困惑度**
   - 法律领域模型通常困惑度在 10-30 之间
   - 如果困惑度 > 50，模型可能未充分训练

2. **BLEU/ROUGE**
   - 法律问答任务，ROUGE-L > 0.4 通常表示模型表现良好
   - BLEU 分数通常较低（0.2-0.5），因为法律文本多样性高

3. **长度匹配**
   - 生成答案长度应与参考答案相近
   - 如果差异过大，可能需要调整生成参数

## 🐛 常见问题

### Q: TensorBoard 显示 "No dashboards are active"
A: 确保训练已经开始，并且日志目录中有文件。检查 `output/logs/` 目录。

### Q: 评估脚本报错 "rouge_score not found"
A: 安装依赖：`pip install rouge-score nltk`

### Q: 评估速度很慢
A: 使用 `--max_samples` 限制样本数，或减少困惑度计算的样本数（在代码中修改）。

### Q: 如何比较不同模型的性能？
A: 分别运行评估脚本，对比 `evaluation_results.json` 中的指标。


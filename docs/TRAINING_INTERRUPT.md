# 训练中断和恢复指南

## 🛑 如何安全中断训练

### 方法 1: 使用 Ctrl+C（推荐）

在运行训练的终端中按 `Ctrl+C`：

```bash
# 训练过程中按 Ctrl+C
^C
```

**注意事项：**
- 第一次按 `Ctrl+C`：发送中断信号，训练会完成当前步骤后停止
- 如果训练没有响应，再按一次 `Ctrl+C`：强制终止（不推荐，可能损坏检查点）

### 方法 2: 查找并终止进程

```bash
# 查找训练进程
ps aux | grep train.py

# 或使用
pgrep -f train.py

# 优雅终止（发送 SIGTERM）
kill <PID>

# 如果不行，强制终止（不推荐）
kill -9 <PID>
```

## ✅ 训练中断后的状态

训练脚本会在以下情况下自动保存检查点：
- 每 `save_steps` 步（配置文件中设置）
- 每个 epoch 结束时
- 训练正常完成时

**检查点位置：** `output/checkpoint-*/`

## 🔄 如何恢复训练

### 方法 1: 从检查点恢复（推荐）

训练脚本会自动从最新的检查点恢复训练：

```bash
# 直接运行训练脚本，会自动检测并恢复
python train.py
```

### 方法 2: 指定检查点路径

如果需要从特定检查点恢复：

```bash
# 修改 train.py 或使用命令行参数（如果支持）
# 或者直接修改配置文件中的 output_dir
```

## 📊 检查训练进度

```bash
# 查看检查点
ls -lh output/checkpoint-*/

# 查看训练日志
tail -f output/logs/training.log

# 查看 TensorBoard
tensorboard --logdir output/logs
```

## ⚠️ 注意事项

1. **不要强制终止（kill -9）**：可能导致检查点损坏
2. **确保有足够的磁盘空间**：检查点文件可能很大
3. **定期备份检查点**：重要的训练可以备份到其他位置
4. **检查 GPU 显存**：中断后确保显存已释放

## 🔧 常见问题

### Q: 中断后显存没有释放？

```bash
# 检查 GPU 进程
nvidia-smi

# 如果有残留进程，手动终止
fuser -v /dev/nvidia*
kill -9 <PID>
```

### Q: 检查点损坏怎么办？

如果检查点损坏，可以从上一个检查点恢复，或者重新开始训练。

### Q: 如何修改参数后继续训练？

1. 修改 `config/train_config.yaml`
2. 确保 `output_dir` 指向同一个目录
3. 运行 `python train.py`，会自动从最新检查点恢复


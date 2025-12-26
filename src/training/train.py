import torch
import yaml
import os
import json
from datetime import datetime
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer

# GPU 监控回调
try:
    from .gpu_monitor import GPUMonitorCallback
except ImportError:
    # 如果相对导入失败，尝试绝对导入
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from gpu_monitor import GPUMonitorCallback

# === 0. 读取配置文件函数 ===
from pathlib import Path

# 获取项目根目录
project_root = Path(__file__).parent.parent.parent

def load_config(config_path=None):
    if config_path is None:
        config_path = project_root / "config" / "train_config.yaml"
    else:
        config_path = project_root / config_path
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

# 加载配置
cfg = load_config()
print(f"Loading configuration from {project_root / 'config' / 'train_config.yaml'}...")

# === 1. 量化配置 (从 Config 读取) ===
bnb_config = None
if cfg['quantization']['load_in_4bit']:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16 if not cfg['training']['bf16'] else torch.bfloat16,
        bnb_4bit_use_double_quant=False,
    )

# === 2. 加载模型与分词器 ===
print(f"Loading model: {cfg['model']['name']}")
model = AutoModelForCausalLM.from_pretrained(
    cfg['model']['name'],
    quantization_config=bnb_config,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(cfg['model']['name'], trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# === 3. 准备 LoRA 配置 ===
peft_config = LoraConfig(
    r=cfg['lora']['r'],
    lora_alpha=cfg['lora']['lora_alpha'],
    lora_dropout=cfg['lora']['lora_dropout'],
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=cfg['lora']['target_modules']
)

# === 4. 加载数据集 ===
print(f"Loading training data from: {cfg['data']['train_path']}")
train_dataset = load_dataset("json", data_files=cfg['data']['train_path'], split="train")

# 加载验证集（如果存在）
eval_dataset = None
if cfg['data'].get('val_path') and Path(cfg['data']['val_path']).exists():
    print(f"Loading validation data from: {cfg['data']['val_path']}")
    eval_dataset = load_dataset("json", data_files=cfg['data']['val_path'], split="train")
else:
    print("⚠️  验证集不存在，将跳过评估")

# === 5. 训练参数设置 ===
# 创建日志目录
logging_dir = cfg['training'].get('logging_dir', os.path.join(cfg['training']['output_dir'], 'logs'))
os.makedirs(logging_dir, exist_ok=True)

# 获取评估和保存参数
eval_steps = cfg['training'].get('eval_steps', 0)
save_steps = cfg['training']['save_steps']
load_best_model_at_end = cfg['training'].get('load_best_model_at_end', False)
do_eval = cfg['training'].get('do_eval', False) and eval_dataset is not None
eval_strategy = cfg['training'].get('eval_strategy', cfg['training'].get('evaluation_strategy', 'no'))

# 如果启用 load_best_model_at_end 且使用 steps 评估策略，需要确保 save_steps 是 eval_steps 的倍数
if load_best_model_at_end and do_eval and eval_strategy == 'steps' and eval_steps > 0:
    if save_steps % eval_steps != 0:
        # 自动调整 save_steps 为 eval_steps 的倍数
        if save_steps < eval_steps:
            # 如果 save_steps 小于 eval_steps，调整为 eval_steps
            adjusted_save_steps = eval_steps
        else:
            # 如果 save_steps 大于 eval_steps，调整为最接近的倍数（向下取整）
            adjusted_save_steps = (save_steps // eval_steps) * eval_steps
            if adjusted_save_steps == 0:
                adjusted_save_steps = eval_steps
        
        print(f"⚠️  自动调整 save_steps: {save_steps} -> {adjusted_save_steps} (必须是 eval_steps={eval_steps} 的倍数)")
        save_steps = adjusted_save_steps

training_args = TrainingArguments(
    output_dir=cfg['training']['output_dir'],
    num_train_epochs=cfg['training']['num_train_epochs'],
    per_device_train_batch_size=cfg['training']['per_device_train_batch_size'],
    per_device_eval_batch_size=cfg['training'].get('per_device_eval_batch_size', cfg['training']['per_device_train_batch_size']),
    gradient_accumulation_steps=cfg['training']['gradient_accumulation_steps'],
    optim=cfg['training']['optim'],
    save_steps=save_steps,
    logging_steps=cfg['training']['logging_steps'],
    eval_steps=eval_steps,
    do_eval=do_eval,
    eval_strategy=eval_strategy,
    load_best_model_at_end=load_best_model_at_end,
    metric_for_best_model=cfg['training'].get('metric_for_best_model', 'eval_loss'),
    greater_is_better=cfg['training'].get('metric_for_best_model', 'eval_loss') != 'eval_loss',
    learning_rate=float(cfg['training']['learning_rate']),
    weight_decay=0.001,
    fp16=cfg['training']['fp16'],
    bf16=cfg['training']['bf16'], # 5090 推荐 True
    max_grad_norm=0.3,
    warmup_ratio=cfg['training']['warmup_ratio'],
    group_by_length=True,
    lr_scheduler_type="constant",
    # 显存优化：启用梯度检查点（以时间换显存）
    gradient_checkpointing=cfg['training'].get('gradient_checkpointing', True),
    # 可视化设置
    report_to=cfg['training'].get('report_to', 'tensorboard'),
    logging_dir=logging_dir,
    # 保存训练历史
    save_total_limit=3,  # 只保留最近 3 个检查点
    logging_first_step=True,
)

# === 6. 启用梯度检查点（如果配置了）===
if cfg['training'].get('gradient_checkpointing', True):
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        print("✅ 已启用梯度检查点（节省显存，训练速度会稍慢）")
    else:
        print("⚠️  模型不支持梯度检查点")

# === 7. 初始化 SFTTrainer 中的自定义数据格式化函数 ===
def formatting_prompts_func(example):
    # 确保 'input' 字段存在，即使是空字符串
    input_text = example.get('input', '')
    # Llama 3 标准对话模板
    text = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{example['instruction']}\n{input_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{example['output']}<|eot_id|>"
    return text

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    args=training_args,
    formatting_func=formatting_prompts_func,
)

# === 添加 GPU 监控回调 ===
if cfg['training'].get('gpu_monitor', {}).get('enabled', True):
    gpu_monitor = GPUMonitorCallback(
        log_interval=cfg['training'].get('gpu_monitor', {}).get('log_interval', 10),
        enable_tensorboard=cfg['training'].get('gpu_monitor', {}).get('enable_tensorboard', True)
    )
    trainer.add_callback(gpu_monitor)
    print("✅ GPU 监控已启用")

# === 7. 开始训练 ===
print("="*60)
print("🚀 开始训练...")
print(f"📊 TensorBoard 日志目录: {logging_dir}")
print(f"   启动 TensorBoard: tensorboard --logdir {logging_dir}")
print("="*60)

# 记录训练开始时间
train_start_time = datetime.now()

try:
    trainer.train()
    train_end_time = datetime.now()
    training_duration = (train_end_time - train_start_time).total_seconds() / 3600  # 转换为小时
    
    print("\n" + "="*60)
    print("✅ 训练完成！")
    print(f"⏱️  训练时长: {training_duration:.2f} 小时")
    print("="*60)
    
    # 保存训练统计信息
    training_stats = {
        "training_start": train_start_time.isoformat(),
        "training_end": train_end_time.isoformat(),
        "training_duration_hours": round(training_duration, 2),
        "total_steps": trainer.state.global_step,
        "total_epochs": trainer.state.epoch,
        "best_metric": trainer.state.best_metric if hasattr(trainer.state, 'best_metric') else None,
        "best_model_checkpoint": trainer.state.best_model_checkpoint if hasattr(trainer.state, 'best_model_checkpoint') else None,
    }
    
    stats_file = os.path.join(cfg['training']['output_dir'], 'training_stats.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(training_stats, f, ensure_ascii=False, indent=2)
    print(f"💾 训练统计已保存到: {stats_file}")
    
except Exception as e:
    print(f"\n❌ 训练过程中出现错误: {e}")
    raise

# === 8. 保存模型 ===
# 保存 LoRA 适配器
final_save_path = os.path.join(cfg['training']['output_dir'], cfg['model']['new_name'])
print(f"\n💾 保存模型到: {final_save_path}...")
trainer.model.save_pretrained(final_save_path)
tokenizer.save_pretrained(final_save_path)
print("✅ 模型保存完成！")

# === 9. 提示评估 ===
print("\n" + "="*60)
print("📊 训练完成，建议进行模型评估：")
print(f"   python src/training/evaluate.py --model_path {final_save_path}")
print("="*60)
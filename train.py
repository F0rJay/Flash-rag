import torch
import yaml
import os
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer

# === 0. 读取配置文件函数 ===
def load_config(config_path="config/train_config.yaml"):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

# 加载配置
cfg = load_config()
print(f"Loading configuration from config/train_config.yaml...")

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
print(f"Loading data from: {cfg['data']['path']}")
dataset = load_dataset("json", data_files=cfg['data']['path'], split="train")

# === 5. 训练参数设置 ===
training_args = TrainingArguments(
    output_dir=cfg['training']['output_dir'],
    num_train_epochs=cfg['training']['num_train_epochs'],
    per_device_train_batch_size=cfg['training']['per_device_train_batch_size'],
    gradient_accumulation_steps=cfg['training']['gradient_accumulation_steps'],
    optim=cfg['training']['optim'],
    save_steps=cfg['training']['save_steps'],
    logging_steps=cfg['training']['logging_steps'],
    learning_rate=float(cfg['training']['learning_rate']),
    weight_decay=0.001,
    fp16=cfg['training']['fp16'],
    bf16=cfg['training']['bf16'], # 5090 推荐 True
    max_grad_norm=0.3,
    warmup_ratio=cfg['training']['warmup_ratio'],
    group_by_length=True,
    lr_scheduler_type="constant",
)

# === 6. 初始化 SFTTrainer 中的自定义数据格式化函数 ===
def formatting_prompts_func(example):
    # 确保 'input' 字段存在，即使是空字符串
    input_text = example.get('input', '')
    # Llama 3 标准对话模板
    text = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{example['instruction']}\n{input_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{example['output']}<|eot_id|>"
    return text

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    args=training_args,
    formatting_func=formatting_prompts_func,
)

# === 7. 开始训练 ===
print("Starting training...")
trainer.train()

# === 8. 保存模型 ===
# 保存 LoRA 适配器
final_save_path = os.path.join(cfg['training']['output_dir'], cfg['model']['new_name'])
print(f"Saving adapter to {final_save_path}...")
trainer.model.save_pretrained(final_save_path)
tokenizer.save_pretrained(final_save_path)
print("Training completed successfully!")
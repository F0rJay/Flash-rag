import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig
from accelerate import Accelerator
import yaml

# --- 0. 加载配置 ---
def load_config(config_path="config/train_config.yaml"):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

cfg = load_config()

# 模型名称和路径
BASE_MODEL_ID = cfg['model']['name']
LORA_ADAPTER_PATH = os.path.join(cfg['training']['output_dir'], cfg['model']['new_name'])
MERGED_MODEL_PATH = os.path.join(cfg['training']['output_dir'], "llama3-law-assistant-merged")

# --- 1. 加载基础模型和适配器 ---
print(f"Loading base model: {BASE_MODEL_ID}...")
# 注意：合并时，我们需要加载 FP16 或 BF16 的完整模型（不再是 4-bit 量化）
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.bfloat16 if cfg['training']['bf16'] else torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)

print(f"Loading LoRA adapter from: {LORA_ADAPTER_PATH}...")
model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)

# --- 2. 合并权重 ---
print("Merging LoRA weights into base model...")
# model.merge_and_unload() 执行关键操作
# 它将 LoRA 矩阵 A 和 B 相乘的结果加回到原始模型权重中
merged_model = model.merge_and_unload()

# --- 3. 保存合并后的模型 ---
print(f"Saving merged model to {MERGED_MODEL_PATH}...")

# 保存为标准的 Hugging Face 格式
merged_model.save_pretrained(MERGED_MODEL_PATH, safe_serialization=True)
tokenizer.save_pretrained(MERGED_MODEL_PATH)
print("Merging completed. The production-ready model is saved!")
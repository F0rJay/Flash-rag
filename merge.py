import torch
import os
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# === 0. 读取配置 ===
def load_config(config_path="config/train_config.yaml"):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

cfg = load_config()

# === 1. 定义路径 ===
# 基础模型路径
base_model_path = cfg['model']['name']
# LoRA 适配器路径 (训练输出目录 + 新模型名)
adapter_path = os.path.join(cfg['training']['output_dir'], cfg['model']['new_name'])
# 最终合并模型保存路径
output_path = os.path.join(cfg['training']['output_dir'], "llama3-law-merged")

print(f"Base Model: {base_model_path}")
print(f"Adapter: {adapter_path}")
print(f"Output: {output_path}")

# === 2. 加载模型 ===
print("Loading base model in BF16/FP16...")
# 注意：合并时必须加载完整精度的模型，不能用 4-bit 量化
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16 if cfg['training']['bf16'] else torch.float16,
    device_map="auto",
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(
    base_model_path, 
    trust_remote_code=True
)

# === 3. 加载并合并 LoRA ===
print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(model, adapter_path)

print("Merging weights (this may take a moment)...")
model = model.merge_and_unload()

# === 4. 保存最终模型 ===
print(f"Saving merged model to {output_path}...")
model.save_pretrained(output_path, safe_serialization=True) # 保存为 safetensors 格式
tokenizer.save_pretrained(output_path)

print("✅ Merge completed! You are ready for vLLM deployment.")
#!/usr/bin/env python3
"""
模型评估脚本
功能：
1. 在测试集上评估模型性能
2. 计算多种评估指标（BLEU、ROUGE、困惑度等）
3. 生成评估报告
4. 保存评估结果

使用方法：
    python src/training/evaluate.py --model_path output/llama3-law-assistant-lora [--test_path data/datasets/test.jsonl]
"""

import torch
import yaml
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel
import os

# 评估指标
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    print("⚠️  警告: rouge_score 未安装，将跳过 ROUGE 指标计算")
    print("   安装: pip install rouge-score")

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.tokenize import word_tokenize
    import nltk
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    BLEU_AVAILABLE = True
except ImportError:
    BLEU_AVAILABLE = False
    print("⚠️  警告: nltk 未安装，将跳过 BLEU 指标计算")
    print("   安装: pip install nltk")

# 获取项目根目录
project_root = Path(__file__).parent.parent.parent

def load_config(config_path=None):
    """加载配置文件"""
    if config_path is None:
        config_path = project_root / "config" / "train_config.yaml"
    else:
        config_path = project_root / config_path
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def load_model_and_tokenizer(model_path, config):
    """加载模型和分词器"""
    print(f"📂 加载模型: {model_path}")
    
    # 检查是否是 LoRA 模型
    if (Path(model_path) / "adapter_config.json").exists():
        print("   检测到 LoRA 适配器，加载基础模型...")
        base_model_name = config['model']['name']
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16 if config['training']['bf16'] else torch.float16,
        )
        model = PeftModel.from_pretrained(base_model, model_path)
        print("   ✅ LoRA 适配器已加载")
    else:
        # 完整模型
        bnb_config = None
        if config['quantization'].get('load_in_4bit', False):
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16 if not config['training']['bf16'] else torch.bfloat16,
            )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16 if config['training']['bf16'] else torch.float16,
        )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    return model, tokenizer

def format_prompt(instruction, input_text=""):
    """格式化提示词（与训练时一致）"""
    return f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{instruction}\n{input_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

def generate_response(model, tokenizer, prompt, max_length=512, temperature=0.7):
    """生成回答"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # 解码输出
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # 提取 assistant 的回答部分
    if "<|start_header_id|>assistant<|end_header_id|>" in generated_text:
        assistant_response = generated_text.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
        assistant_response = assistant_response.split("<|eot_id|>")[0].strip()
    else:
        # 如果没有找到标记，返回生成的部分（去除输入）
        assistant_response = generated_text[len(prompt):].strip()
    
    return assistant_response

def calculate_perplexity(model, tokenizer, texts):
    """计算困惑度"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
            labels = inputs["input_ids"]
            
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            total_loss += loss.item() * labels.numel()
            total_tokens += labels.numel()
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = np.exp(avg_loss)
    return perplexity

def calculate_bleu(references, predictions):
    """计算 BLEU 分数"""
    if not BLEU_AVAILABLE:
        return None
    
    smoothing = SmoothingFunction().method1
    bleu_scores = []
    
    for ref, pred in zip(references, predictions):
        ref_tokens = word_tokenize(ref.lower())
        pred_tokens = word_tokenize(pred.lower())
        score = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothing)
        bleu_scores.append(score)
    
    return {
        'bleu_1': np.mean(bleu_scores),
        'bleu_avg': np.mean(bleu_scores)
    }

def calculate_rouge(references, predictions):
    """计算 ROUGE 分数"""
    if not ROUGE_AVAILABLE:
        return None
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    
    for ref, pred in zip(references, predictions):
        scores = scorer.score(ref, pred)
        rouge_scores['rouge1'].append(scores['rouge1'].fmeasure)
        rouge_scores['rouge2'].append(scores['rouge2'].fmeasure)
        rouge_scores['rougeL'].append(scores['rougeL'].fmeasure)
    
    return {
        'rouge1': np.mean(rouge_scores['rouge1']),
        'rouge2': np.mean(rouge_scores['rouge2']),
        'rougeL': np.mean(rouge_scores['rougeL'])
    }

def evaluate_model(model, tokenizer, test_dataset, config, max_samples=None):
    """评估模型"""
    print(f"\n📊 开始评估模型...")
    print(f"   测试集大小: {len(test_dataset)}")
    if max_samples:
        test_dataset = test_dataset.select(range(min(max_samples, len(test_dataset))))
        print(f"   评估样本数: {len(test_dataset)} (限制)")
    
    references = []
    predictions = []
    all_texts = []  # 用于计算困惑度
    
    print("\n🔄 生成预测...")
    for idx, example in enumerate(test_dataset):
        if (idx + 1) % 100 == 0:
            print(f"   进度: {idx + 1}/{len(test_dataset)}")
        
        instruction = example.get('instruction', '')
        input_text = example.get('input', '')
        reference = example.get('output', '')
        
        # 格式化提示词
        prompt = format_prompt(instruction, input_text)
        
        # 生成回答
        prediction = generate_response(model, tokenizer, prompt)
        
        references.append(reference)
        predictions.append(prediction)
        
        # 用于困惑度计算
        full_text = prompt + reference
        all_texts.append(full_text)
    
    print("\n📈 计算评估指标...")
    
    # 计算困惑度
    print("   计算困惑度...")
    perplexity = calculate_perplexity(model, tokenizer, all_texts[:100])  # 限制样本数以加快速度
    
    # 计算 BLEU
    bleu_scores = None
    if BLEU_AVAILABLE:
        print("   计算 BLEU 分数...")
        bleu_scores = calculate_bleu(references, predictions)
    
    # 计算 ROUGE
    rouge_scores = None
    if ROUGE_AVAILABLE:
        print("   计算 ROUGE 分数...")
        rouge_scores = calculate_rouge(references, predictions)
    
    # 计算平均长度
    avg_ref_length = np.mean([len(ref) for ref in references])
    avg_pred_length = np.mean([len(pred) for pred in predictions])
    
    # 汇总结果
    results = {
        'evaluation_time': datetime.now().isoformat(),
        'num_samples': len(test_dataset),
        'perplexity': float(perplexity) if perplexity else None,
        'bleu': bleu_scores,
        'rouge': rouge_scores,
        'average_lengths': {
            'reference': float(avg_ref_length),
            'prediction': float(avg_pred_length)
        },
        'sample_predictions': [
            {
                'instruction': test_dataset[i]['instruction'][:100] + '...' if len(test_dataset[i]['instruction']) > 100 else test_dataset[i]['instruction'],
                'reference': references[i][:200] + '...' if len(references[i]) > 200 else references[i],
                'prediction': predictions[i][:200] + '...' if len(predictions[i]) > 200 else predictions[i],
            }
            for i in range(min(5, len(test_dataset)))
        ]
    }
    
    return results

def print_results(results):
    """打印评估结果"""
    print("\n" + "="*60)
    print("📊 模型评估结果")
    print("="*60)
    print(f"评估时间: {results['evaluation_time']}")
    print(f"评估样本数: {results['num_samples']}")
    print()
    
    if results['perplexity']:
        print(f"困惑度 (Perplexity): {results['perplexity']:.2f}")
    
    if results['bleu']:
        print(f"\nBLEU 分数:")
        print(f"  BLEU-1: {results['bleu']['bleu_1']:.4f}")
        print(f"  BLEU-Avg: {results['bleu']['bleu_avg']:.4f}")
    
    if results['rouge']:
        print(f"\nROUGE 分数:")
        print(f"  ROUGE-1: {results['rouge']['rouge1']:.4f}")
        print(f"  ROUGE-2: {results['rouge']['rouge2']:.4f}")
        print(f"  ROUGE-L: {results['rouge']['rougeL']:.4f}")
    
    print(f"\n平均长度:")
    print(f"  参考答案: {results['average_lengths']['reference']:.1f} 字符")
    print(f"  生成答案: {results['average_lengths']['prediction']:.1f} 字符")
    
    print(f"\n示例预测 (前 3 个):")
    for i, sample in enumerate(results['sample_predictions'][:3], 1):
        print(f"\n示例 {i}:")
        print(f"  问题: {sample['instruction']}")
        print(f"  参考答案: {sample['reference']}")
        print(f"  模型预测: {sample['prediction']}")

def main():
    parser = argparse.ArgumentParser(description='评估训练后的模型')
    parser.add_argument('--model_path', type=str, required=True,
                       help='模型路径（LoRA 适配器或完整模型）')
    parser.add_argument('--test_path', type=str, default=None,
                       help='测试集路径（默认: config 中的 test_path）')
    parser.add_argument('--config', type=str, default=None,
                       help='配置文件路径（默认: config/train_config.yaml）')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='最大评估样本数（用于快速测试）')
    parser.add_argument('--output', type=str, default=None,
                       help='评估结果保存路径（默认: model_path/evaluation_results.json）')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 确定测试集路径
    if args.test_path:
        test_path = Path(args.test_path)
    else:
        test_path = project_root / config['data'].get('test_path', 'data/datasets/test.jsonl')
    
    if not test_path.exists():
        print(f"❌ 错误: 测试集不存在: {test_path}")
        return
    
    # 确定模型路径
    model_path = Path(args.model_path)
    if not model_path.is_absolute():
        model_path = project_root / model_path
    
    if not model_path.exists():
        print(f"❌ 错误: 模型路径不存在: {model_path}")
        return
    
    # 加载模型和分词器
    model, tokenizer = load_model_and_tokenizer(str(model_path), config)
    
    # 加载测试集
    print(f"\n📂 加载测试集: {test_path}")
    test_dataset = load_dataset("json", data_files=str(test_path), split="train")
    
    # 评估模型
    results = evaluate_model(model, tokenizer, test_dataset, config, args.max_samples)
    
    # 打印结果
    print_results(results)
    
    # 保存结果
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = model_path / "evaluation_results.json"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 评估结果已保存到: {output_path}")
    print("="*60)

if __name__ == "__main__":
    main()


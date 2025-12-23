#!/usr/bin/env python3
"""
数据集分析和验证脚本
功能：
1. 验证数据集格式是否正确
2. 统计数据集基本信息（数量、长度分布等）
3. 检查数据质量（空值、重复等）
4. 生成数据集报告

使用方法：
    python scripts/analyze_dataset.py [--train data/datasets/train.jsonl] [--val data/datasets/val.jsonl] [--test data/datasets/test.jsonl]
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from collections import Counter
import statistics

def load_jsonl(file_path: Path) -> List[Dict]:
    """加载 JSONL 文件"""
    data = []
    errors = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                errors.append((line_num, str(e)))
    return data, errors

def validate_format(item: Dict, required_fields: List[str] = None) -> tuple:
    """
    验证数据格式
    返回: (is_valid, error_message)
    """
    if required_fields is None:
        required_fields = ['instruction', 'input', 'output']
    
    # 检查必需字段
    for field in required_fields:
        if field not in item:
            return False, f"缺少必需字段: {field}"
        if not isinstance(item[field], str):
            return False, f"字段 {field} 必须是字符串类型"
    
    # 检查 instruction 和 output 不能为空
    if not item['instruction'].strip():
        return False, "instruction 字段不能为空"
    if not item['output'].strip():
        return False, "output 字段不能为空"
    
    return True, ""

def analyze_dataset(data: List[Dict], dataset_name: str) -> Dict:
    """分析数据集"""
    stats = {
        'name': dataset_name,
        'total': len(data),
        'valid': 0,
        'invalid': 0,
        'errors': [],
        'length_stats': {},
        'field_stats': {}
    }
    
    # 统计字段长度
    instruction_lengths = []
    input_lengths = []
    output_lengths = []
    total_lengths = []
    
    # 验证数据
    for idx, item in enumerate(data):
        is_valid, error_msg = validate_format(item)
        if is_valid:
            stats['valid'] += 1
            # 统计长度
            inst_len = len(item['instruction'])
            input_len = len(item.get('input', ''))
            output_len = len(item['output'])
            total_len = inst_len + input_len + output_len
            
            instruction_lengths.append(inst_len)
            input_lengths.append(input_len)
            output_lengths.append(output_len)
            total_lengths.append(total_len)
        else:
            stats['invalid'] += 1
            stats['errors'].append({
                'index': idx,
                'error': error_msg,
                'item': {k: v[:100] if isinstance(v, str) and len(v) > 100 else v 
                        for k, v in item.items()}
            })
    
    # 计算长度统计
    if instruction_lengths:
        stats['length_stats']['instruction'] = {
            'min': min(instruction_lengths),
            'max': max(instruction_lengths),
            'mean': round(statistics.mean(instruction_lengths), 2),
            'median': statistics.median(instruction_lengths)
        }
    
    if input_lengths:
        stats['length_stats']['input'] = {
            'min': min(input_lengths),
            'max': max(input_lengths),
            'mean': round(statistics.mean(input_lengths), 2),
            'median': statistics.median(input_lengths)
        }
    
    if output_lengths:
        stats['length_stats']['output'] = {
            'min': min(output_lengths),
            'max': max(output_lengths),
            'mean': round(statistics.mean(output_lengths), 2),
            'median': statistics.median(output_lengths)
        }
    
    if total_lengths:
        stats['length_stats']['total'] = {
            'min': min(total_lengths),
            'max': max(total_lengths),
            'mean': round(statistics.mean(total_lengths), 2),
            'median': statistics.median(total_lengths)
        }
    
    # 检查重复
    unique_instructions = set()
    duplicates = []
    for idx, item in enumerate(data):
        inst = item.get('instruction', '').strip()
        if inst in unique_instructions:
            duplicates.append(idx)
        else:
            unique_instructions.add(inst)
    
    stats['duplicates'] = len(duplicates)
    stats['unique_instructions'] = len(unique_instructions)
    
    return stats

def print_stats(stats: Dict):
    """打印统计信息"""
    print(f"\n{'='*60}")
    print(f"📊 数据集分析: {stats['name']}")
    print(f"{'='*60}")
    print(f"总数量: {stats['total']}")
    print(f"✅ 有效: {stats['valid']}")
    print(f"❌ 无效: {stats['invalid']}")
    print(f"🔄 重复指令: {stats['duplicates']}")
    print(f"✨ 唯一指令: {stats['unique_instructions']}")
    
    if stats['length_stats']:
        print(f"\n📏 长度统计:")
        for field, lengths in stats['length_stats'].items():
            print(f"  {field}:")
            print(f"    最小: {lengths['min']}")
            print(f"    最大: {lengths['max']}")
            print(f"    平均: {lengths['mean']}")
            print(f"    中位数: {lengths['median']}")
    
    if stats['errors']:
        print(f"\n⚠️  错误示例 (前 5 个):")
        for error in stats['errors'][:5]:
            print(f"  索引 {error['index']}: {error['error']}")
            print(f"    数据: {error['item']}")

def main():
    parser = argparse.ArgumentParser(description='分析和验证数据集')
    parser.add_argument('--train', type=str, default='data/datasets/train.jsonl',
                       help='训练集路径（默认: data/datasets/train.jsonl）')
    parser.add_argument('--val', type=str, default='data/datasets/val.jsonl',
                       help='验证集路径（默认: data/datasets/val.jsonl）')
    parser.add_argument('--test', type=str, default='data/datasets/test.jsonl',
                       help='测试集路径（默认: data/datasets/test.jsonl）')
    parser.add_argument('--output', type=str, default=None,
                       help='输出报告文件路径（可选，JSON 格式）')
    
    args = parser.parse_args()
    
    # 获取项目根目录
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    all_stats = {}
    
    # 分析每个数据集
    datasets = [
        ('train', args.train),
        ('val', args.val),
        ('test', args.test)
    ]
    
    for name, file_path in datasets:
        file_path = Path(file_path)
        if not file_path.is_absolute():
            file_path = project_root / file_path
        
        if not file_path.exists():
            print(f"⚠️  警告: {name} 数据集不存在: {file_path}")
            continue
        
        print(f"\n📂 加载 {name} 数据集: {file_path}")
        data, errors = load_jsonl(file_path)
        
        if errors:
            print(f"⚠️  警告: 发现 {len(errors)} 个 JSON 解析错误")
            for line_num, error in errors[:3]:
                print(f"  第 {line_num} 行: {error}")
        
        stats = analyze_dataset(data, name)
        all_stats[name] = stats
        print_stats(stats)
    
    # 汇总统计
    if len(all_stats) > 1:
        print(f"\n{'='*60}")
        print(f"📈 汇总统计")
        print(f"{'='*60}")
        total_samples = sum(s['total'] for s in all_stats.values())
        total_valid = sum(s['valid'] for s in all_stats.values())
        total_invalid = sum(s['invalid'] for s in all_stats.values())
        
        print(f"总样本数: {total_samples}")
        print(f"总有效数: {total_valid}")
        print(f"总无效数: {total_invalid}")
        print(f"有效率: {total_valid/total_samples*100:.2f}%")
        
        print(f"\n各数据集分布:")
        for name, stats in all_stats.items():
            ratio = stats['total'] / total_samples * 100 if total_samples > 0 else 0
            print(f"  {name}: {stats['total']} ({ratio:.1f}%)")
    
    # 保存报告
    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = project_root / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_stats, f, ensure_ascii=False, indent=2)
        print(f"\n💾 报告已保存到: {output_path}")

if __name__ == "__main__":
    main()


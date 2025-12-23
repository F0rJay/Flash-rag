#!/usr/bin/env python3
"""
数据集准备脚本
功能：
1. 将 DISC-Law 格式转换为项目需要的格式
2. 将数据集分成训练集、验证集、测试集
3. 保存到 data/datasets/ 目录

使用方法：
    python scripts/prepare_dataset.py <input_file> [--train-ratio 0.8] [--val-ratio 0.1] [--test-ratio 0.1]
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List
import random

def load_jsonl(file_path: Path) -> List[Dict]:
    """加载 JSONL 文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def convert_format(item: Dict) -> Dict:
    """
    转换数据格式
    从 DISC-Law 格式: {"id": "...", "input": "...", "output": "..."}
    转换为项目格式: {"instruction": "...", "input": "...", "output": "..."}
    """
    # DISC-Law 格式：input 是问题，output 是答案
    # 项目格式：instruction 是问题，input 是上下文（可为空），output 是答案
    
    converted = {
        "instruction": item.get("input", ""),  # 问题作为 instruction
        "input": "",  # 通常为空，如果有上下文可以填充
        "output": item.get("output", "")  # 答案
    }
    
    return converted

def split_dataset(data: List[Dict], train_ratio: float, val_ratio: float, test_ratio: float) -> tuple:
    """
    划分数据集
    返回: (train_data, val_data, test_data)
    """
    # 验证比例
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例之和必须等于 1.0"
    
    # 打乱数据
    random.seed(42)  # 固定随机种子，确保可复现
    shuffled_data = data.copy()
    random.shuffle(shuffled_data)
    
    total = len(shuffled_data)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    train_data = shuffled_data[:train_end]
    val_data = shuffled_data[train_end:val_end]
    test_data = shuffled_data[val_end:]
    
    return train_data, val_data, test_data

def save_jsonl(data: List[Dict], file_path: Path):
    """保存为 JSONL 格式"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def main():
    parser = argparse.ArgumentParser(description='准备训练数据集')
    parser.add_argument('input_file', type=str, nargs='?', help='输入的 JSONL 文件路径（可选，如果已有 train/val/test 文件则不需要）')
    parser.add_argument('--train-ratio', type=float, default=0.8, help='训练集比例（默认: 0.8）')
    parser.add_argument('--val-ratio', type=float, default=0.1, help='验证集比例（默认: 0.1）')
    parser.add_argument('--test-ratio', type=float, default=0.1, help='测试集比例（默认: 0.1）')
    parser.add_argument('--use-existing', action='store_true', help='直接使用已有的 train/val/test.jsonl 文件，不进行转换和划分')
    parser.add_argument('--validate', action='store_true', help='验证已有数据集格式')
    
    args = parser.parse_args()
    
    # 获取项目根目录
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    output_dir = project_root / "data" / "datasets"
    
    # 如果使用已有数据集
    if args.use_existing or args.validate:
        train_file = output_dir / "train.jsonl"
        val_file = output_dir / "val.jsonl"
        test_file = output_dir / "test.jsonl"
        
        existing_files = []
        for name, file_path in [("训练集", train_file), ("验证集", val_file), ("测试集", test_file)]:
            if file_path.exists():
                existing_files.append((name, file_path))
                print(f"✅ {name}存在: {file_path}")
            else:
                print(f"⚠️  {name}不存在: {file_path}")
        
        if args.validate:
            # 验证数据集
            print("\n🔍 验证数据集格式...")
            import json
            all_valid = True
            for name, file_path in existing_files:
                data = []
                errors = []
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        if not line.strip():
                            continue
                        try:
                            item = json.loads(line)
                            data.append(item)
                            # 简单验证
                            if 'instruction' not in item or 'output' not in item:
                                errors.append(f"缺少必需字段")
                            elif not item.get('instruction', '').strip() or not item.get('output', '').strip():
                                errors.append(f"字段为空")
                        except json.JSONDecodeError as e:
                            errors.append(f"JSON解析错误: {e}")
                
                if errors:
                    print(f"❌ {name} 有 {len(errors)} 个错误")
                    all_valid = False
                else:
                    print(f"✅ {name} 格式正确，共 {len(data)} 条数据")
            
            if all_valid:
                print("\n✅ 所有数据集验证通过！")
            return
        
        if args.use_existing:
            print(f"\n✅ 使用已有数据集，无需转换")
            return
    
    # 如果没有提供输入文件，提示用户
    if not args.input_file:
        print("❌ 错误: 请提供输入文件路径，或使用 --use-existing 使用已有数据集")
        print("使用方法:")
        print("  1. 从新文件转换: python scripts/prepare_dataset.py <input_file>")
        print("  2. 使用已有数据集: python scripts/prepare_dataset.py --use-existing")
        print("  3. 验证数据集: python scripts/prepare_dataset.py --validate")
        return
    
    # 输入文件路径
    input_file = Path(args.input_file)
    if not input_file.is_absolute():
        # 尝试相对于项目根目录的父目录（autodl-tmp）
        if (project_root.parent / input_file).exists():
            input_file = project_root.parent / input_file
        # 或者相对于当前工作目录
        elif Path(input_file).exists():
            input_file = Path(input_file).resolve()
        else:
            input_file = project_root / input_file
    
    if not input_file.exists():
        print(f"❌ 错误: 文件不存在: {input_file}")
        return
    
    print(f"📂 加载数据文件: {input_file}")
    
    # 加载数据
    raw_data = load_jsonl(input_file)
    print(f"✅ 加载了 {len(raw_data)} 条数据")
    
    # 转换格式
    print("🔄 转换数据格式...")
    converted_data = [convert_format(item) for item in raw_data]
    print(f"✅ 转换完成")
    
    # 划分数据集
    print(f"📊 划分数据集 (训练:{args.train_ratio}, 验证:{args.val_ratio}, 测试:{args.test_ratio})...")
    train_data, val_data, test_data = split_dataset(
        converted_data, 
        args.train_ratio, 
        args.val_ratio, 
        args.test_ratio
    )
    
    print(f"✅ 划分完成:")
    print(f"   - 训练集: {len(train_data)} 条")
    print(f"   - 验证集: {len(val_data)} 条")
    print(f"   - 测试集: {len(test_data)} 条")
    
    # 保存文件
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_file = output_dir / "train.jsonl"
    val_file = output_dir / "val.jsonl"
    test_file = output_dir / "test.jsonl"
    
    print(f"💾 保存文件...")
    save_jsonl(train_data, train_file)
    save_jsonl(val_data, val_file)
    save_jsonl(test_data, test_file)
    
    print(f"✅ 保存完成:")
    print(f"   - 训练集: {train_file}")
    print(f"   - 验证集: {val_file}")
    print(f"   - 测试集: {test_file}")
    
    # 显示示例
    print("\n📝 数据格式示例（训练集第一条）:")
    print(json.dumps(train_data[0], ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()


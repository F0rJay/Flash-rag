#!/usr/bin/env python3
"""
RAG 知识库准备脚本
功能：
1. 从 DISC-Law JSONL 文件中提取内容
   - 模式1: 提取法律条文（reference 字段）- 法条型 RAG
   - 模式2: 提取案例（input + output）- 案例型 RAG
   - 模式3: 提取判决书（input 字段）- 判决书型 RAG
   - 模式4: 提取法条+案例（混合模式）
2. 去重并合并
3. 保存为文本文件供 ingest.py 使用

使用方法：
    # 提取法律条文（法条型 RAG）
    python scripts/prepare_rag_knowledge.py <file1.jsonl> --mode law --output data/docs/legal_docs.txt
    
    # 提取案例（案例型 RAG）
    python scripts/prepare_rag_knowledge.py <file1.jsonl> --mode case --output data/docs/case_docs.txt
    
    # 提取判决书（判决书型 RAG）
    python scripts/prepare_rag_knowledge.py <file1.jsonl> --mode judgement --output data/docs/judgement_docs.txt
    
    # 混合模式（法条+案例）
    python scripts/prepare_rag_knowledge.py <file1.jsonl> --mode mixed --output data/docs/mixed_docs.txt
"""

import json
import argparse
from pathlib import Path
from typing import Set, List

def extract_references_from_jsonl(file_path: Path) -> Set[str]:
    """从 JSONL 文件中提取所有 reference 字段的内容（法条）"""
    references = set()
    
    print(f"📂 处理文件: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        count = 0
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                # 提取 reference 字段（可能是列表）
                if 'reference' in item:
                    refs = item['reference']
                    if isinstance(refs, list):
                        for ref in refs:
                            if ref and ref.strip():
                                references.add(ref.strip())
                    elif isinstance(refs, str) and refs.strip():
                        references.add(refs.strip())
                count += 1
            except json.JSONDecodeError as e:
                print(f"⚠️  警告: 第 {line_num} 行 JSON 解析失败: {e}")
                continue
    
    print(f"✅ 从 {count} 条记录中提取了 {len(references)} 条唯一法律条文")
    return references

def extract_cases_from_jsonl(file_path: Path) -> List[str]:
    """从 JSONL 文件中提取案例（input + output）"""
    cases = []
    
    print(f"📂 处理文件: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        count = 0
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                # 提取案例：案件描述 + 判决结果
                input_text = item.get('input', '').strip()
                output_text = item.get('output', '').strip()
                
                if input_text and output_text:
                    # 格式化案例文本
                    case_text = f"【案件事实】\n{input_text}\n\n【判决结果】\n{output_text}"
                    cases.append(case_text)
                count += 1
            except json.JSONDecodeError as e:
                print(f"⚠️  警告: 第 {line_num} 行 JSON 解析失败: {e}")
                continue
    
    print(f"✅ 从 {count} 条记录中提取了 {len(cases)} 个案例")
    return cases

def extract_judgements_from_jsonl(file_path: Path) -> List[str]:
    """从 JSONL 文件中提取判决书（input 字段，包含完整判决书原文）"""
    judgements = []
    
    print(f"📂 处理文件: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        count = 0
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                # 提取判决书：完整的判决书原文（input 字段）
                input_text = item.get('input', '').strip()
                output_text = item.get('output', '').strip()  # 摘要，可选
                
                if input_text:
                    # 如果 input 包含"请大致描述"等提示词，去除
                    if input_text.startswith("请大致描述") or input_text.startswith("这是一篇法律文书"):
                        # 找到第一个换行后的内容
                        lines = input_text.split('\n', 1)
                        if len(lines) > 1:
                            input_text = lines[1].strip()
                    
                    # 格式化判决书文本（包含摘要）
                    if output_text:
                        judgement_text = f"【判决书摘要】\n{output_text}\n\n【判决书全文】\n{input_text}"
                    else:
                        judgement_text = f"【判决书全文】\n{input_text}"
                    judgements.append(judgement_text)
                count += 1
            except json.JSONDecodeError as e:
                print(f"⚠️  警告: 第 {line_num} 行 JSON 解析失败: {e}")
                continue
    
    print(f"✅ 从 {count} 条记录中提取了 {len(judgements)} 份判决书")
    return judgements

def merge_references(file_paths: List[Path]) -> List[str]:
    """合并多个文件中的法律条文并去重"""
    all_references = set()
    
    for file_path in file_paths:
        if not file_path.exists():
            print(f"❌ 错误: 文件不存在: {file_path}")
            continue
        
        refs = extract_references_from_jsonl(file_path)
        all_references.update(refs)
    
    # 转换为列表并排序（便于阅读）
    sorted_references = sorted(list(all_references))
    return sorted_references

def merge_cases(file_paths: List[Path]) -> List[str]:
    """合并多个文件中的案例"""
    all_cases = []
    
    for file_path in file_paths:
        if not file_path.exists():
            print(f"❌ 错误: 文件不存在: {file_path}")
            continue
        
        cases = extract_cases_from_jsonl(file_path)
        all_cases.extend(cases)
    
    return all_cases

def extract_mixed(file_paths: List[Path]) -> List[str]:
    """混合模式：提取法条和案例"""
    mixed_content = []
    
    # 提取法条
    references = merge_references(file_paths)
    mixed_content.extend([f"【法律条文】\n{ref}" for ref in references])
    
    # 提取案例
    cases = merge_cases(file_paths)
    mixed_content.extend(cases)
    
    return mixed_content

def merge_judgements(file_paths: List[Path]) -> List[str]:
    """合并多个文件中的判决书"""
    all_judgements = []
    
    for file_path in file_paths:
        if not file_path.exists():
            print(f"❌ 错误: 文件不存在: {file_path}")
            continue
        
        judgements = extract_judgements_from_jsonl(file_path)
        all_judgements.extend(judgements)
    
    return all_judgements

def save_to_text(content: List[str], output_path: Path, content_type: str = "法律条文"):
    """保存内容到文本文件"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in content:
            f.write(item + '\n\n')  # 每个条目之间空两行
    
    print(f"💾 已保存 {len(content)} 条{content_type}到: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='准备 RAG 知识库（从 JSONL 提取内容）')
    parser.add_argument('files', nargs='+', type=str, help='输入的 JSONL 文件路径（可多个）')
    parser.add_argument('--mode', type=str, choices=['law', 'case', 'judgement', 'mixed'], default='law',
                       help='提取模式: law=法律条文, case=案例, judgement=判决书, mixed=混合（默认: law）')
    parser.add_argument('--output', type=str, default=None,
                       help='输出文件路径（默认根据模式自动生成）')
    
    args = parser.parse_args()
    
    # 获取项目根目录
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # 处理输入文件路径
    input_files = []
    for file_path in args.files:
        file_path = Path(file_path)
        if not file_path.is_absolute():
            # 尝试相对于项目根目录的父目录
            if (project_root.parent / file_path).exists():
                file_path = project_root.parent / file_path
            elif (project_root / file_path).exists():
                file_path = project_root / file_path
            else:
                file_path = Path(file_path).resolve()
        input_files.append(file_path)
    
    # 处理输出文件路径
    if args.output is None:
        # 根据模式自动生成输出路径
        mode_map = {
            'law': 'data/docs/legal_docs.txt',
            'case': 'data/docs/case_docs.txt',
            'judgement': 'data/docs/judgement_docs.txt',
            'mixed': 'data/docs/mixed_docs.txt'
        }
        output_path = project_root / mode_map[args.mode]
    else:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = project_root / output_path
    
    print("=" * 60)
    print("📚 RAG 知识库准备工具")
    print("=" * 60)
    print(f"模式: {args.mode}")
    print()
    
    # 根据模式提取内容
    if args.mode == 'law':
        content = merge_references(input_files)
        content_type = "法律条文"
    elif args.mode == 'case':
        content = merge_cases(input_files)
        content_type = "案例"
    elif args.mode == 'judgement':
        content = merge_judgements(input_files)
        content_type = "判决书"
    else:  # mixed
        content = extract_mixed(input_files)
        content_type = "混合内容（法条+案例）"
    
    if not content:
        print(f"❌ 错误: 未能提取到任何{content_type}")
        return
    
    # 保存到文件
    save_to_text(content, output_path, content_type)
    
    # 显示统计信息
    print()
    print("=" * 60)
    print("📊 统计信息")
    print("=" * 60)
    print(f"输入文件数: {len(input_files)}")
    print(f"提取的{content_type}数: {len(content)}")
    print(f"输出文件: {output_path}")
    if output_path.exists():
        print(f"文件大小: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    print()
    print("📝 示例（前 2 条）:")
    for i, item in enumerate(content[:2], 1):
        preview = item[:150] + "..." if len(item) > 150 else item
        print(f"\n{i}. {preview}")

if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
Query Rewrite (查询改写) 模块
功能：将用户的口语化问题改写为专业的法律检索关键词
提升检索准确率，特别是在法律术语匹配方面
"""

import os
from typing import Optional
from pathlib import Path
import sys

# 设置 HuggingFace 镜像环境变量
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.CustomVLLM import CustomVLLM


class QueryRewriter:
    """查询改写器，使用 LLM 将用户问题改写为专业检索关键词"""
    
    def __init__(self, llm: Optional[CustomVLLM] = None, vllm_url: str = "http://localhost:8000"):
        """
        初始化查询改写器
        
        Args:
            llm: CustomVLLM 实例，如果为 None 则自动创建
            vllm_url: vLLM 服务地址
        """
        if llm is None:
            self.llm = CustomVLLM(base_url=vllm_url)
        else:
            self.llm = llm
        
        # 查询改写提示词模板
        self.rewrite_prompt_template = """你是一个专业的法律检索助手。请将用户的问题改写为适合法律知识库检索的专业关键词或短语。

改写要求：
1. 保留原问题的核心法律概念
2. 将口语化表达转换为法律术语
3. 提取关键的法律实体和关系
4. 保持简洁，通常不超过20个字
5. 如果是法律条文查询，保留具体的法律名称和条款关键词

示例：
- 用户问题："他不还钱咋办？"
- 改写结果："债务违约 违约责任 还款义务"

- 用户问题："合同到期了还能续签吗？"
- 改写结果："合同续签 合同期限 续约"

- 用户问题："工伤怎么赔偿？"
- 改写结果："工伤赔偿 工伤保险 工伤认定"

现在请改写以下问题：

用户问题：{query}

改写结果（只输出改写后的关键词，不要其他解释）："""
    
    def rewrite(self, query: str, max_retries: int = 2) -> str:
        """
        改写用户查询
        
        Args:
            query: 原始用户查询
            max_retries: 最大重试次数（如果改写失败，返回原查询）
            
        Returns:
            改写后的查询关键词
        """
        if not query or not query.strip():
            return query
        
        # 构建提示词
        prompt = self.rewrite_prompt_template.format(query=query)
        
        # 尝试调用 LLM 进行改写
        for attempt in range(max_retries + 1):
            try:
                # 调用 LLM
                response = self.llm(prompt)
                
                # 清理响应（去除可能的引号、换行等）
                rewritten = response.strip()
                rewritten = rewritten.strip('"').strip("'").strip()
                
                # 如果响应为空或太短，返回原查询
                if not rewritten or len(rewritten) < 3:
                    if attempt < max_retries:
                        continue
                    return query
                
                # 如果响应太长，可能是 LLM 输出了额外内容，尝试提取前部分
                if len(rewritten) > 100:
                    # 尝试提取第一行或前50个字符
                    lines = rewritten.split('\n')
                    if lines:
                        rewritten = lines[0].strip()
                    if len(rewritten) > 100:
                        rewritten = rewritten[:100]
                
                print(f"📝 查询改写: '{query}' -> '{rewritten}'")
                return rewritten
                
            except Exception as e:
                print(f"⚠️  查询改写失败 (尝试 {attempt + 1}/{max_retries + 1}): {e}")
                if attempt < max_retries:
                    continue
                # 所有重试都失败，返回原查询
                print(f"⚠️  查询改写失败，使用原查询: '{query}'")
                return query
        
        return query
    
    def rewrite_batch(self, queries: list, max_retries: int = 2) -> list:
        """
        批量改写查询
        
        Args:
            queries: 查询列表
            max_retries: 最大重试次数
            
        Returns:
            改写后的查询列表
        """
        return [self.rewrite(query, max_retries) for query in queries]


def create_query_rewriter(llm: Optional[CustomVLLM] = None, vllm_url: str = "http://localhost:8000") -> QueryRewriter:
    """
    创建查询改写器实例（工厂函数）
    
    Args:
        llm: CustomVLLM 实例
        vllm_url: vLLM 服务地址
        
    Returns:
        QueryRewriter 实例
    """
    return QueryRewriter(llm=llm, vllm_url=vllm_url)


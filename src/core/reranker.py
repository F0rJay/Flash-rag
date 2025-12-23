#!/usr/bin/env python3
"""
Rerank (重排序) 模块
功能：使用 Cross-Encoder 模型对检索结果进行精细重排序
提升检索精度，特别是在法律术语等专业领域
"""

import os
import torch
from typing import List, Dict, Tuple
from pathlib import Path

# 设置 HuggingFace 镜像环境变量
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

try:
    from sentence_transformers import CrossEncoder
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("⚠️  警告: sentence-transformers 未安装，Rerank 功能将不可用")
    print("   安装: pip install sentence-transformers")


class Reranker:
    """重排序器，使用 Cross-Encoder 模型对检索结果进行精细排序"""
    
    def __init__(self, model_name: str = "BAAI/bge-reranker-base", device: str = None):
        """
        初始化重排序器
        
        Args:
            model_name: Cross-Encoder 模型名称，默认使用 BGE-Reranker
            device: 设备（'cuda' 或 'cpu'），None 表示自动选择
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers 未安装，请运行: pip install sentence-transformers")
        
        self.model_name = model_name
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"🔄 加载 Rerank 模型: {model_name}")
        try:
            self.model = CrossEncoder(model_name, device=self.device)
            print(f"✅ Rerank 模型加载成功 (设备: {self.device})")
        except Exception as e:
            print(f"❌ Rerank 模型加载失败: {e}")
            print(f"   尝试使用备用模型...")
            # 备用模型
            try:
                self.model = CrossEncoder("ms-marco-MiniLM-L-6-v2", device=self.device)
                self.model_name = "ms-marco-MiniLM-L-6-v2"
                print(f"✅ 备用 Rerank 模型加载成功")
            except Exception as e2:
                raise RuntimeError(f"无法加载任何 Rerank 模型: {e2}")
    
    def rerank(
        self, 
        query: str, 
        documents: List[str], 
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        对文档进行重排序
        
        Args:
            query: 查询文本
            documents: 文档列表（从向量检索得到的 Top K 文档）
            top_k: 返回前 K 个结果
            
        Returns:
            List[Tuple[str, float]]: 排序后的文档和分数列表，按分数降序排列
        """
        if not documents:
            return []
        
        # 构建 query-document 对
        pairs = [[query, doc] for doc in documents]
        
        # 使用 Cross-Encoder 进行打分
        scores = self.model.predict(pairs)
        
        # 将分数和文档配对，并按分数降序排序
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # 返回 Top K
        return scored_docs[:top_k]
    
    def rerank_with_metadata(
        self,
        query: str,
        documents_with_metadata: List[Dict],
        top_k: int = 5
    ) -> List[Dict]:
        """
        对带元数据的文档进行重排序
        
        Args:
            query: 查询文本
            documents_with_metadata: 文档字典列表，每个字典包含 'page_content' 和可能的其他元数据
            top_k: 返回前 K 个结果
            
        Returns:
            List[Dict]: 排序后的文档字典列表，每个字典包含 'page_content', 'score' 和原始元数据
        """
        if not documents_with_metadata:
            return []
        
        # 提取文档内容
        documents = [doc.get('page_content', doc.get('content', str(doc))) for doc in documents_with_metadata]
        
        # 重排序
        scored_docs = self.rerank(query, documents, top_k)
        
        # 构建结果，保留原始元数据
        results = []
        for doc_content, score in scored_docs:
            # 找到对应的原始文档
            original_doc = next(
                (d for d in documents_with_metadata 
                 if d.get('page_content', d.get('content', str(d))) == doc_content),
                None
            )
            
            if original_doc:
                result = original_doc.copy()
                result['score'] = float(score)
                results.append(result)
            else:
                # 如果找不到原始文档，创建新字典
                results.append({
                    'page_content': doc_content,
                    'score': float(score)
                })
        
        return results


def create_reranker(model_name: str = "BAAI/bge-reranker-base", device: str = None) -> Reranker:
    """
    创建重排序器实例（工厂函数）
    
    Args:
        model_name: Cross-Encoder 模型名称
        device: 设备
        
    Returns:
        Reranker 实例
    """
    return Reranker(model_name=model_name, device=device)


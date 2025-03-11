"""检索模块

这个模块负责根据用户查询检索最相关的文本块。
"""

from typing import List, Dict, Any, Optional, Union

import numpy as np

from .embeddings import TextEmbedder, EmbedderFactory
from .vector_store import VectorStore


class Retriever:
    """检索器基类"""
    
    def __init__(self, vector_store: VectorStore, embedder: TextEmbedder):
        """初始化检索器
        
        Args:
            vector_store: 向量存储实例
            embedder: 文本嵌入器实例
        """
        self.vector_store = vector_store
        self.embedder = embedder
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """检索与查询最相关的文档
        
        Args:
            query: 用户查询
            top_k: 返回的最相似文档数量
            
        Returns:
            最相似的文档列表，按相似度降序排列
        """
        raise NotImplementedError("子类必须实现retrieve方法")


class VectorRetriever(Retriever):
    """基于向量相似度的检索器"""
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """检索与查询最相关的文档
        
        Args:
            query: 用户查询
            top_k: 返回的最相似文档数量
            
        Returns:
            最相似的文档列表，按相似度降序排列
        """
        # 将查询转换为向量
        query_vector = self.embedder.embed_text(query)
        
        # 在向量存储中搜索最相似的文档
        results = self.vector_store.search(query_vector, top_k=top_k)
        
        return results


class HybridRetriever(Retriever):
    """混合检索器，结合向量检索和关键词匹配"""
    
    def __init__(self, vector_store: VectorStore, embedder: TextEmbedder, 
                 keyword_weight: float = 0.3):
        """初始化混合检索器
        
        Args:
            vector_store: 向量存储实例
            embedder: 文本嵌入器实例
            keyword_weight: 关键词匹配的权重（0-1之间）
        """
        super().__init__(vector_store, embedder)
        self.keyword_weight = keyword_weight
    
    def _keyword_match(self, query: str, documents: List[Dict[str, Any]]) -> List[float]:
        if not documents:
            return []
            
        scores = []
        query_words = set(query.lower().split())
        for doc in documents:
            content_words = set(doc['content'].lower().split())
            intersection = query_words & content_words
            scores.append(len(intersection) / len(query_words) if query_words else 0.0)
        return scores

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        try:
            vector_results = super().retrieve(query, top_k)
            keyword_scores = self._keyword_match(query, self.vector_store.documents)
            
            combined_results = []
            for i, vec_result in enumerate(vector_results):
                combined_score = (1 - self.keyword_weight) * vec_result['score']
                if i < len(keyword_scores):
                    combined_score += self.keyword_weight * keyword_scores[i]
                vec_result['combined_score'] = combined_score
                combined_results.append(vec_result)
            
            return sorted(combined_results, key=lambda x: x['combined_score'], reverse=True)[:top_k]
        except Exception as e:
            raise RuntimeError(f"混合检索失败: {str(e)}") from e


class RetrieverFactory:
    """检索器工厂类"""
    
    @staticmethod
    def get_retriever(retriever_type: str, vector_store: VectorStore, 
                      embedder_type: str = 'sentence_transformer', **kwargs) -> Retriever:
        """根据类型返回相应的检索器
        
        Args:
            retriever_type: 检索器类型，可选值为 'vector', 'hybrid'
            vector_store: 向量存储实例
            embedder_type: 嵌入器类型
            **kwargs: 传递给检索器和嵌入器构造函数的参数
            
        Returns:
            检索器实例
        """
        # 创建嵌入器
        embedder = EmbedderFactory.get_embedder(embedder_type, **kwargs)
        
        # 创建检索器
        if retriever_type == 'vector':
            return VectorRetriever(vector_store, embedder)
        elif retriever_type == 'hybrid':
            keyword_weight = kwargs.get('keyword_weight', 0.3)
            return HybridRetriever(vector_store, embedder, keyword_weight=keyword_weight)
        else:
            raise ValueError(f"不支持的检索器类型: {retriever_type}")
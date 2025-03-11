"""向量嵌入模块

这个模块负责将文本转换为向量表示，以便于后续的相似度检索。
"""

import os
from typing import List, Dict, Any, Optional, Union

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class TextEmbedder:
    """文本嵌入器基类"""
    
    def __init__(self):
        pass
    
    def embed_text(self, text: str) -> np.ndarray:
        """将单个文本转换为向量"""
        raise NotImplementedError("子类必须实现embed_text方法")
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """将多个文本转换为向量"""
        raise NotImplementedError("子类必须实现embed_texts方法")


class SentenceTransformerEmbedder(TextEmbedder):
    """基于SentenceTransformer的文本嵌入器"""
    
    def __init__(self, model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2'):
        """初始化SentenceTransformer嵌入器
        
        Args:
            model_name: SentenceTransformer模型名称
        """
        super().__init__()
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
    
    def embed_text(self, text: str) -> np.ndarray:
        """将单个文本转换为向量
        
        Args:
            text: 输入文本
            
        Returns:
            文本的向量表示
        """
        return self.model.encode(text)
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """将多个文本转换为向量
        
        Args:
            texts: 输入文本列表
            
        Returns:
            文本的向量表示列表
        """
        return self.model.encode(texts)


class OpenAIEmbedder(TextEmbedder):
    """基于OpenAI API的文本嵌入器"""
    
    def __init__(self, model_name: str = 'text-embedding-ada-002', api_key: Optional[str] = None):
        """初始化OpenAI嵌入器
        
        Args:
            model_name: OpenAI模型名称
            api_key: OpenAI API密钥，如果为None则从环境变量获取
        """
        super().__init__()
        import openai
        
        self.model_name = model_name
        if api_key:
            openai.api_key = api_key
        elif 'OPENAI_API_KEY' in os.environ:
            openai.api_key = os.environ['OPENAI_API_KEY']
        else:
            raise ValueError("必须提供OpenAI API密钥或设置OPENAI_API_KEY环境变量")
        
        self.client = openai.OpenAI()
    
    def embed_text(self, text: str) -> np.ndarray:
        """将单个文本转换为向量"""
        response = self.client.embeddings.create(
            input=text,
            model=self.model_name
        )
        return np.array(response.data[0].embedding)
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """将多个文本转换为向量"""
        response = self.client.embeddings.create(
            input=texts,
            model=self.model_name
        )
        return np.array([data.embedding for data in response.data])


class EmbedderFactory:
    """嵌入器工厂类"""
    
    @staticmethod
    def get_embedder(embedder_type: str, **kwargs) -> TextEmbedder:
        """根据类型返回相应的嵌入器
        
        Args:
            embedder_type: 嵌入器类型，支持'sentence_transformer'、'openai'和'qwen'
            **kwargs: 传递给嵌入器的参数
            
        Returns:
            文本嵌入器实例
        """
        if embedder_type == 'sentence_transformer':
            model_name = kwargs.get('model_name', 'paraphrase-multilingual-MiniLM-L12-v2')
            return SentenceTransformerEmbedder(model_name=model_name)
        elif embedder_type == 'openai':
            model_name = kwargs.get('model_name', 'text-embedding-ada-002')
            api_key = kwargs.get('api_key', None)
            return OpenAIEmbedder(model_name=model_name, api_key=api_key)
        elif embedder_type == 'qwen':
            # 导入QwenEmbedder
            from .qwen_api import QwenEmbedder, DASHSCOPE_AVAILABLE
            if not DASHSCOPE_AVAILABLE:
                raise ImportError("DashScope未安装，请先安装dashscope库")
            model_name = kwargs.get('model_name', 'text-embedding-v2')
            api_key = kwargs.get('api_key', None)
            return QwenEmbedder(model_name=model_name, api_key=api_key)
        else:
            raise ValueError(f"不支持的嵌入器类型: {embedder_type}")


def generate_embeddings(chunks: List[Dict[str, Any]], embedder_type: str = 'sentence_transformer', **kwargs) -> List[Dict[str, Any]]:
    """
    为文本块生成嵌入向量
    
    Args:
        chunks: 文本块列表，每个块是一个字典，包含'text'和'metadata'字段
        embedder_type: 嵌入器类型
        **kwargs: 传递给嵌入器的参数
        
    Returns:
        添加了嵌入向量的文本块列表
    """
    # 获取嵌入器
    embedder = EmbedderFactory.get_embedder(embedder_type, **kwargs)
    
    # 提取文本内容
    texts = [chunk['text'] for chunk in chunks]
    
    # 生成嵌入向量
    print(f"使用 {embedder_type} 生成 {len(texts)} 个文本块的嵌入向量...")
    embeddings = embedder.embed_texts(texts)
    
    # 将嵌入向量添加到文本块中
    for i, chunk in enumerate(chunks):
        chunk['embedding'] = embeddings[i]
    
    return chunks
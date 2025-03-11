"""RAG流程集成模块

这个模块整合了RAG系统的所有组件，提供端到端的检索增强生成功能。
"""

import os
import json
from typing import List, Dict, Any, Optional, Union

from .document_loader import load_documents
from .text_chunker import split_documents, ChunkerFactory
from .embeddings import generate_embeddings, EmbedderFactory
from .vector_store import VectorStore, SimpleVectorStore, FAISSVectorStore, ChromaVectorStore
from .retriever import RetrieverFactory
from .generator import GeneratorFactory


class RAGPipeline:
    """RAG系统流程类"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """初始化RAG流程
        
        Args:
            config: 配置字典，包含各组件的参数
        """
        self.config = config or {}
        
        # 设置默认配置
        self._set_default_config()
        
        # 初始化向量存储
        self.vector_store = self._init_vector_store()
        
        # 初始化检索器
        self.retriever = RetrieverFactory.get_retriever(
            retriever_type=self.config['retriever']['type'],
            vector_store=self.vector_store,
            embedder_type=self.config['embedder']['type'],
            **self.config['embedder']['params']
        )
        
        # 初始化生成器
        self.generator = GeneratorFactory.get_generator(
            generator_type=self.config['generator']['type'],
            **self.config['generator']['params']
        )
    
    def _set_default_config(self) -> None:
        """设置默认配置"""
        default_config = {
            'chunker': {
                'type': 'paragraph',
                'params': {
                    'chunk_size': 1000,
                    'chunk_overlap': 200
                }
            },
            'embedder': {
                'type': 'sentence_transformer',
                'params': {
                    'model_name': 'paraphrase-multilingual-MiniLM-L12-v2'
                }
            },
            'vector_store': {
                'type': 'simple',
                'params': {}
            },
            'retriever': {
                'type': 'vector',
                'params': {
                    'top_k': 5
                }
            },
            'generator': {
                'type': 'template',
                'params': {}
            }
        }
        
        # 合并用户配置和默认配置
        for key, default_value in default_config.items():
            if key not in self.config:
                self.config[key] = default_value
            elif isinstance(default_value, dict):
                for subkey, subvalue in default_value.items():
                    if subkey not in self.config[key]:
                        self.config[key][subkey] = subvalue
    
    def _init_vector_store(self) -> VectorStore:
        """初始化向量存储"""
        vector_store_type = self.config['vector_store']['type']
        params = self.config['vector_store']['params']
        
        if vector_store_type == 'simple':
            return SimpleVectorStore()
        elif vector_store_type == 'faiss':
            dimension = params.get('dimension', None)
            return FAISSVectorStore(dimension=dimension)
        elif vector_store_type == 'chroma':
            collection_name = params.get('collection_name', 'rag_documents')
            persist_directory = params.get('persist_directory', None)
            return ChromaVectorStore(collection_name=collection_name, persist_directory=persist_directory)
        else:
            raise ValueError(f"不支持的向量存储类型: {vector_store_type}")
    
    def index_documents(self, documents_dir: str) -> None:
        """索引文档目录
        
        Args:
            documents_dir: 文档目录路径
        """
        print(f"开始索引文档目录: {documents_dir}")
        
        # 加载文档
        print("1. 加载文档...")
        documents = load_documents(documents_dir)
        print(f"   加载了 {len(documents)} 个文档")
        
        # 文本分块
        print("2. 文本分块...")
        # 将文档列表转换为字典格式，以适应split_documents函数的要求
        documents_dict = {doc['metadata']['source']: doc['content'] for doc in documents}
        chunks = split_documents(
            documents_dict,
            chunker_type=self.config['chunker']['type'],
            **self.config['chunker']['params']
        )
        print(f"   生成了 {len(chunks)} 个文本块")
        
        # 生成嵌入向量
        print("3. 生成嵌入向量...")
        chunks_with_embeddings = generate_embeddings(
            chunks,
            embedder_type=self.config['embedder']['type'],
            **self.config['embedder']['params']
        )
        
        # 添加到向量存储
        print("4. 添加到向量存储...")
        self.vector_store.add_documents(chunks_with_embeddings)
        print("   索引完成")
    
    def save(self, directory: str) -> None:
        """保存RAG系统状态
        
        Args:
            directory: 保存目录路径
        """
        os.makedirs(directory, exist_ok=True)
        
        # 保存向量存储
        vector_store_dir = os.path.join(directory, "vector_store")
        self.vector_store.save(vector_store_dir)
        
        # 保存配置
        config_path = os.path.join(directory, "config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, ensure_ascii=False, indent=2)
        
        print(f"RAG系统已保存到: {directory}")
    
    def load(self, directory: str) -> None:
        """加载RAG系统状态
        
        Args:
            directory: 加载目录路径
        """
        # 加载配置
        config_path = os.path.join(directory, "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
            self._set_default_config()  # 确保所有必要的配置都存在
        
        # 重新初始化向量存储
        self.vector_store = self._init_vector_store()
        
        # 加载向量存储
        vector_store_dir = os.path.join(directory, "vector_store")
        self.vector_store.load(vector_store_dir)
        
        # 重新初始化检索器和生成器
        self.retriever = RetrieverFactory.get_retriever(
            retriever_type=self.config['retriever']['type'],
            vector_store=self.vector_store,
            embedder_type=self.config['embedder']['type'],
            **self.config['embedder']['params']
        )
        
        self.generator = GeneratorFactory.get_generator(
            generator_type=self.config['generator']['type'],
            **self.config['generator']['params']
        )
        
        print(f"RAG系统已从 {directory} 加载")
    
    def query(self, query: str, top_k: int = None) -> str:
        """处理用户查询
        
        Args:
            query: 用户查询
            top_k: 检索的文档数量，如果为None则使用配置中的值
            
        Returns:
            生成的回答
        """
        if top_k is None:
            top_k = self.config['retriever']['params'].get('top_k', 5)
        
        # 检索相关文档
        contexts = self.retriever.retrieve(query, top_k=top_k)
        
        # 生成回答
        answer = self.generator.generate(query, contexts)
        
        return answer


def create_rag_pipeline(config_path: str = None) -> RAGPipeline:
    """创建RAG流程实例
    
    Args:
        config_path: 配置文件路径，如果为None则使用默认配置
        
    Returns:
        RAG流程实例
    """
    config = None
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    
    return RAGPipeline(config=config)
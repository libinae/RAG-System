"""向量存储模块

这个模块负责存储和检索文本块的向量表示。
"""

import os
import pickle
from typing import List, Dict, Any, Optional, Union, Tuple

import numpy as np
from tqdm import tqdm

# 尝试导入FAISS，如果不可用则提供警告
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("警告: FAISS未安装，将无法使用FAISSVectorStore")

# 尝试导入Chroma，如果不可用则提供警告
try:
    import chromadb
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    print("警告: ChromaDB未安装，将无法使用ChromaVectorStore")


class VectorStore:
    """向量存储基类"""
    
    def __init__(self):
        pass
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """添加文档到向量存储"""
        raise NotImplementedError("子类必须实现add_documents方法")
    
    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """搜索最相似的文档"""
        raise NotImplementedError("子类必须实现search方法")
    
    def save(self, path: str) -> None:
        """保存向量存储到磁盘"""
        raise NotImplementedError("子类必须实现save方法")
    
    def load(self, path: str) -> None:
        """从磁盘加载向量存储"""
        raise NotImplementedError("子类必须实现load方法")


class SimpleVectorStore(VectorStore):
    """简单的向量存储实现，使用numpy进行向量相似度计算"""
    
    def __init__(self):
        super().__init__()
        self.documents = []
        self.vectors = None
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """添加文档到向量存储
        
        Args:
            documents: 文档列表，每个文档是一个字典，包含'content'、'metadata'和'embedding'字段
        """
        if not documents:
            return
            
        # 添加文档
        self.documents.extend(documents)
        
        # 更新向量矩阵
        vectors = [doc['embedding'] for doc in documents]
        if self.vectors is None:
            self.vectors = np.array(vectors)
        else:
            self.vectors = np.vstack([self.vectors, np.array(vectors)])
    
    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """搜索最相似的文档
        
        Args:
            query_vector: 查询向量
            top_k: 返回的最相似文档数量
            
        Returns:
            最相似的文档列表
        """
        if self.vectors is None or len(self.documents) == 0:
            return []
        
        # 计算余弦相似度
        query_vector = query_vector / np.linalg.norm(query_vector)
        normalized_vectors = self.vectors / np.linalg.norm(self.vectors, axis=1, keepdims=True)
        similarities = np.dot(normalized_vectors, query_vector)
        
        # 获取最相似的文档索引
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # 返回最相似的文档和相似度分数
        results = []
        for idx in top_indices:
            doc = self.documents[idx].copy()
            doc['score'] = float(similarities[idx])
            results.append(doc)
        
        return results
    
    def save(self, path: str) -> None:
        """保存向量存储到磁盘
        
        Args:
            path: 保存目录路径
        """
        try:
            os.makedirs(path, exist_ok=True)
            
            # 保存文档和向量
            with open(os.path.join(path, "documents.pkl"), 'wb') as f:
                pickle.dump(self.documents, f)
            
            if self.vectors is not None:
                with open(os.path.join(path, "vectors.npy"), 'wb') as f:
                    np.save(f, self.vectors)
        except (IOError, PermissionError) as e:
            raise RuntimeError(f"保存向量存储失败: {str(e)}") from e

    def load(self, path: str) -> None:
        try:
            # 加载文档
            documents_path = os.path.join(path, "documents.pkl")
            if os.path.exists(documents_path):
                with open(documents_path, 'rb') as f:
                    self.documents = pickle.load(f)
            
            # 加载向量
            vectors_path = os.path.join(path, "vectors.npy")
            if os.path.exists(vectors_path):
                with open(vectors_path, 'rb') as f:
                    self.vectors = np.load(f)
        except (FileNotFoundError, pickle.UnpicklingError) as e:
            raise RuntimeError(f"加载向量存储失败: {str(e)}") from e


class FAISSVectorStore(VectorStore):
    """基于FAISS的向量存储实现"""
    
    def __init__(self, dimension: Optional[int] = None):
        """初始化FAISS向量存储
        
        Args:
            dimension: 向量维度，如果为None则在添加第一个文档时自动确定
        """
        super().__init__()
        
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS未安装，请先安装FAISS库")
        
        self.dimension = dimension
        self.index = None
        self.documents = []
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """添加文档到向量存储"""
        if not documents:
            return
            
        # 获取向量维度
        if self.dimension is None:
            self.dimension = len(documents[0]['embedding'])
            # 创建FAISS索引
            self.index = faiss.IndexFlatIP(self.dimension)  # 内积索引（余弦相似度）
        
        # 添加文档
        self.documents.extend(documents)
        
        # 添加向量到FAISS索引
        vectors = np.array([doc['embedding'] for doc in documents], dtype=np.float32)
        # 归一化向量，以便使用内积计算余弦相似度
        faiss.normalize_L2(vectors)  
        self.index.add(vectors)
    
    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """搜索最相似的文档"""
        if self.index is None or len(self.documents) == 0:
            return []
        
        # 将查询向量转换为numpy数组并归一化
        query_vector = np.array([query_vector], dtype=np.float32)
        faiss.normalize_L2(query_vector)
        
        # 搜索最相似的向量
        scores, indices = self.index.search(query_vector, top_k)
        
        # 返回最相似的文档和相似度分数
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents):  # 确保索引有效
                doc = self.documents[idx].copy()
                doc['score'] = float(scores[0][i])
                results.append(doc)
        
        return results
    
    def save(self, path: str) -> None:
        """保存向量存储到磁盘"""
        os.makedirs(path, exist_ok=True)
        
        # 保存文档
        with open(os.path.join(path, "documents.pkl"), 'wb') as f:
            pickle.dump(self.documents, f)
        
        # 保存FAISS索引
        if self.index is not None:
            faiss.write_index(self.index, os.path.join(path, "faiss.index"))
            
        # 保存维度信息
        with open(os.path.join(path, "dimension.txt"), 'w') as f:
            f.write(str(self.dimension))
    
    def load(self, path: str) -> None:
        """从磁盘加载向量存储"""
        # 加载文档
        documents_path = os.path.join(path, "documents.pkl")
        if os.path.exists(documents_path):
            with open(documents_path, 'rb') as f:
                self.documents = pickle.load(f)
        
        # 加载维度信息
        dimension_path = os.path.join(path, "dimension.txt")
        if os.path.exists(dimension_path):
            with open(dimension_path, 'r') as f:
                self.dimension = int(f.read().strip())
        
        # 加载FAISS索引
        index_path = os.path.join(path, "faiss.index")
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)


class ChromaVectorStore(VectorStore):
    """基于ChromaDB的向量存储实现"""
    
    def __init__(self, collection_name: str = "rag_documents", persist_directory: Optional[str] = None):
        """初始化ChromaDB向量存储
        
        Args:
            collection_name: 集合名称
            persist_directory: 持久化目录，如果为None则使用内存存储
        """
        super().__init__()
        
        if not CHROMA_AVAILABLE:
            raise ImportError("ChromaDB未安装，请先安装chromadb库")
        
        # 创建ChromaDB客户端
        if persist_directory:
            self.client = chromadb.PersistentClient(path=persist_directory)
        else:
            self.client = chromadb.Client()
        
        # 获取或创建集合
        self.collection = self.client.get_or_create_collection(name=collection_name)
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        # 文档映射，用于存储完整的文档信息
        self.documents = {}
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """添加文档到向量存储"""
        if not documents:
            return
        
        # 准备ChromaDB所需的数据格式
        ids = []
        embeddings = []
        metadatas = []
        contents = []
        
        for i, doc in enumerate(documents):
            # 生成唯一ID
            doc_id = f"doc_{len(self.documents) + i}"
            
            # 添加到文档映射
            self.documents[doc_id] = doc
            
            # 准备ChromaDB数据
            ids.append(doc_id)
            embeddings.append(doc['embedding'].tolist())
            metadatas.append(doc['metadata'])
            contents.append(doc['content'])
        
        # 添加到ChromaDB集合
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=contents
        )
    
    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """搜索最相似的文档"""
        if len(self.documents) == 0:
            return []
        
        # 使用向量搜索
        results = self.collection.query(
            query_embeddings=[query_vector.tolist()],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        # 处理结果
        docs = []
        for i, doc_id in enumerate(results['ids'][0]):
            if doc_id in self.documents:
                doc = self.documents[doc_id].copy()
                # ChromaDB返回的是距离，转换为相似度分数
                doc['score'] = 1.0 - results['distances'][0][i]
                docs.append(doc)
        
        return docs
    
    def save(self, path: str) -> None:
        """保存向量存储到磁盘"""
        os.makedirs(path, exist_ok=True)
        
        # 如果使用持久化存储，ChromaDB已经自动保存了
        # 我们只需要保存文档映射
        with open(os.path.join(path, "documents.pkl"), 'wb') as f:
            pickle.dump(self.documents, f)
        
        # 保存配置信息
        config = {
            'collection_name': self.collection_name,
            'persist_directory': self.persist_directory or os.path.join(path, "chroma_db")
        }
        with open(os.path.join(path, "config.pkl"), 'wb') as f:
            pickle.dump(config, f)
        
        # 如果是内存存储，则需要持久化到磁盘
        if not self.persist_directory:
            self.client.persist()
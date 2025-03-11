#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RAG系统测试脚本

这个脚本用于测试RAG系统的各个组件和整体功能。
"""

import os
import sys
import time
import json
from pathlib import Path

# 添加项目根目录到系统路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.document_loader import load_documents
from src.text_chunker import split_documents, ChunkerFactory
from src.embeddings import generate_embeddings, EmbedderFactory
from src.vector_store import SimpleVectorStore, FAISSVectorStore, ChromaVectorStore
from src.retriever import RetrieverFactory
from src.generator import GeneratorFactory
from src.rag_pipeline import RAGPipeline, create_rag_pipeline


def test_document_loading():
    """测试文档加载功能"""
    print("\n===== 测试文档加载 =====")
    data_dir = os.path.join(project_root, "data")
    documents = load_documents(data_dir)
    
    print(f"加载了 {len(documents)} 个文档")
    for i, doc in enumerate(documents):
        print(f"文档 {i+1}:")
        print(f"  - 文件名: {doc['metadata']['source']}")
        print(f"  - 内容长度: {len(doc['content'])} 字符")
        print(f"  - 内容预览: {doc['content'][:100]}...")
    
    return documents


def test_text_chunking(documents):
    """测试文本分块功能"""
    print("\n===== 测试文本分块 =====")
    
    # 测试不同的分块方法
    chunker_types = ["paragraph", "fixed_size"]
    chunk_sizes = [500, 1000]
    
    results = {}
    
    # 将文档列表转换为字典格式，以适应split_documents函数的要求
    # 优化：直接创建字典，避免在循环中多次赋值
    documents_dict = {doc['metadata']['source']: doc['content'] for doc in documents}
    
    # 打印文档数量，帮助诊断
    print(f"处理 {len(documents_dict)} 个文档，总字符数: {sum(len(content) for content in documents_dict.values())}")
    
    for chunker_type in chunker_types:
        for chunk_size in chunk_sizes:
            print(f"\n使用 {chunker_type} 分块器，块大小 {chunk_size}:")
            
            # 分块
            start_time = time.time()
            chunks = split_documents(
                documents_dict,
                chunker_type=chunker_type,
                chunk_size=chunk_size,
                chunk_overlap=100
            )
            end_time = time.time()
            
            print(f"  - 生成了 {len(chunks)} 个文本块")
            print(f"  - 处理时间: {end_time - start_time:.4f} 秒")
            
            # 显示一些块的信息
            if chunks:
                print(f"  - 第一个块大小: {len(chunks[0]['text'])} 字符")
                print(f"  - 第一个块内容预览: {chunks[0]['text'][:100]}...")
            
            results[(chunker_type, chunk_size)] = chunks
    
    # 返回段落分块器、块大小1000的结果用于后续测试
    return results.get(("paragraph", 1000), [])


def test_embeddings(chunks):
    """测试向量嵌入功能"""
    print("\n===== 测试向量嵌入 =====")
    
    # 使用sentence_transformer嵌入器
    embedder_type = "sentence_transformer"
    model_name = "paraphrase-multilingual-MiniLM-L12-v2"
    
    print(f"使用 {embedder_type} 嵌入器，模型 {model_name}:")
    
    # 创建嵌入器
    embedder = EmbedderFactory.get_embedder(
        embedder_type=embedder_type,
        model_name=model_name
    )
    
    # 生成嵌入向量
    start_time = time.time()
    chunks_with_embeddings = embedder.generate_embeddings(chunks)
    end_time = time.time()
    
    print(f"  - 为 {len(chunks_with_embeddings)} 个文本块生成了嵌入向量")
    print(f"  - 处理时间: {end_time - start_time:.4f} 秒")
    
    # 显示嵌入向量的维度
    if chunks_with_embeddings:
        embedding_dim = len(chunks_with_embeddings[0]['embedding'])
        print(f"  - 嵌入向量维度: {embedding_dim}")
    
    return chunks_with_embeddings


def test_vector_store(chunks_with_embeddings):
    # 增加ChromaDB测试
    try:
        import chromadb
        print("\n测试 ChromaVectorStore:")
        chroma_store = ChromaVectorStore(persist_directory=os.path.join(project_root, "examples", "test_chroma_db"))
        
        start_time = time.time()
        chroma_store.add_documents(chunks_with_embeddings)
        end_time = time.time()
        
        print(f"  - 添加了 {len(chunks_with_embeddings)} 个文档到ChromaDB")
        print(f"  - 处理时间: {end_time - start_time:.4f} 秒")
        
        # 测试搜索
        if chunks_with_embeddings:
            query_vector = chunks_with_embeddings[0]['embedding']
            results = chroma_store.search(query_vector, top_k=3)
            assert len(results) == 3, "ChromaDB搜索结果数量异常"
            
    except ImportError:
        print("ChromaDB未安装，跳过ChromaVectorStore测试")

    # 增加异常情况测试
    print("\n测试异常情况:")
    try:
        invalid_store = SimpleVectorStore()
        invalid_store.load("invalid_path")
    except RuntimeError as e:
        print(f"  - 成功捕获加载异常: {str(e)}")

    # 验证保存目录权限
    try:
        read_only_dir = os.path.join(project_root, "examples", "read_only_dir")
        os.makedirs(read_only_dir, exist_ok=True)
        os.chmod(read_only_dir, 0o444)  # 设置为只读
        
        store = SimpleVectorStore()
        store.save(read_only_dir)
    except PermissionError as e:
        print(f"  - 成功捕获权限异常: {str(e)}")
    finally:
        os.chmod(read_only_dir, 0o755)
    
    # 测试SimpleVectorStore
    print("\n测试 SimpleVectorStore:")
    simple_store = SimpleVectorStore()
    
    # 添加文档
    start_time = time.time()
    simple_store.add_documents(chunks_with_embeddings)
    end_time = time.time()
    
    print(f"  - 添加了 {len(chunks_with_embeddings)} 个文档到向量存储")
    print(f"  - 处理时间: {end_time - start_time:.4f} 秒")
    
    # 测试搜索
    if chunks_with_embeddings:
        query_vector = chunks_with_embeddings[0]['embedding']  # 使用第一个文档的向量作为查询
        results = simple_store.search(query_vector, top_k=3)
        
        print(f"  - 搜索结果数量: {len(results)}")
        for i, result in enumerate(results):
            print(f"    结果 {i+1}: 相似度分数 = {result['score']:.4f}")
            print(f"    内容预览: {result['content'][:100]}...")
    
    # 测试保存和加载
    save_dir = os.path.join(project_root, "examples", "test_vector_store")
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存
    simple_store.save(save_dir)
    print(f"  - 向量存储已保存到 {save_dir}")
    
    # 加载
    new_store = SimpleVectorStore()
    new_store.load(save_dir)
    print(f"  - 从 {save_dir} 加载了向量存储")
    print(f"  - 加载的文档数量: {len(new_store.documents)}")
    
    # 如果有FAISS可用，也测试FAISSVectorStore
    try:
        import faiss
        print("\n测试 FAISSVectorStore:")
        faiss_store = FAISSVectorStore()
        
        # 添加文档
        start_time = time.time()
        faiss_store.add_documents(chunks_with_embeddings)
        end_time = time.time()
        
        print(f"  - 添加了 {len(chunks_with_embeddings)} 个文档到FAISS向量存储")
        print(f"  - 处理时间: {end_time - start_time:.4f} 秒")
        
        # 测试搜索
        if chunks_with_embeddings:
            query_vector = chunks_with_embeddings[0]['embedding']
            results = faiss_store.search(query_vector, top_k=3)
            
            print(f"  - 搜索结果数量: {len(results)}")
            for i, result in enumerate(results):
                print(f"    结果 {i+1}: 相似度分数 = {result['score']:.4f}")
    except ImportError:
        print("FAISS未安装，跳过FAISSVectorStore测试")
    
    return simple_store


def test_retriever(vector_store, chunks_with_embeddings):
    """测试检索器功能"""
    print("\n===== 测试检索器 =====")
    
    # 创建检索器
    retriever = RetrieverFactory.get_retriever(
        retriever_type="vector",
        vector_store=vector_store,
        embedder_type="sentence_transformer",
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )
    
    # 测试检索
    test_queries = [
        "什么是人工智能？",
        "大语言模型有哪些局限性？",
        "RAG系统如何工作？"
    ]
    
    for query in test_queries:
        print(f"\n查询: '{query}'")
        
        start_time = time.time()
        results = retriever.retrieve(query, top_k=3)
        end_time = time.time()
        
        print(f"  - 检索时间: {end_time - start_time:.4f} 秒")
        print(f"  - 检索结果数量: {len(results)}")
        
        for i, result in enumerate(results):
            print(f"    结果 {i+1}: 相似度分数 = {result['score']:.4f}")
            print(f"    内容预览: {result['content'][:100]}...")
    
    return retriever


def test_generator(retriever):
    """测试生成器功能"""
    print("\n===== 测试生成器 =====")
    
    # 创建模板生成器
    generator = GeneratorFactory.get_generator(generator_type="template")
    
    # 测试生成
    test_queries = [
        "什么是人工智能？",
        "大语言模型有哪些局限性？",
        "RAG系统如何工作？"
    ]
    
    for query in test_queries:
        print(f"\n查询: '{query}'")
        
        # 检索相关文档
        retrieved_docs = retriever.retrieve(query, top_k=3)
        
        # 生成回答
        start_time = time.time()
        answer = generator.generate(query, retrieved_docs)
        end_time = time.time()
        
        print(f"  - 生成时间: {end_time - start_time:.4f} 秒")
        print(f"  - 生成的回答:\n{answer}")


def test_rag_pipeline():
    """测试完整的RAG流程"""
    print("\n===== 测试完整RAG流程 =====")
    
    # 创建RAG流程
    config_path = os.path.join(project_root, "config.json")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    rag = RAGPipeline(config)
    
    # 索引文档
    data_dir = os.path.join(project_root, "data")
    print(f"\n索引文档目录: {data_dir}")
    
    start_time = time.time()
    rag.index_documents(data_dir)
    end_time = time.time()
    
    print(f"  - 索引时间: {end_time - start_time:.4f} 秒")
    
    # 测试查询
    test_queries = [
        "什么是人工智能？",
        "大语言模型有哪些局限性？",
        "RAG系统如何工作？"
    ]
    
    for query in test_queries:
        print(f"\n查询: '{query}'")
        
        start_time = time.time()
        answer = rag.query(query)
        end_time = time.time()
        
        print(f"  - 查询时间: {end_time - start_time:.4f} 秒")
        print(f"  - 生成的回答:\n{answer}")
    
    # 测试保存和加载
    save_dir = os.path.join(project_root, "examples", "test_rag_pipeline")
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存
    rag.save(save_dir)
    print(f"\nRAG流程已保存到 {save_dir}")
    
    # 加载
    new_rag = create_rag_pipeline()
    new_rag.load(save_dir)
    print(f"从 {save_dir} 加载了RAG流程")
    
    # 测试加载后的查询
    query = "RAG系统的优势是什么？"
    print(f"\n加载后查询: '{query}'")
    
    answer = new_rag.query(query)
    print(f"  - 生成的回答:\n{answer}")


def main():
    """主函数"""
    print("开始测试RAG系统...\n")
    
    # 测试各个组件
    documents = test_document_loading()
    chunks = test_text_chunking(documents)
    chunks_with_embeddings = test_embeddings(chunks)
    vector_store = test_vector_store(chunks_with_embeddings)
    retriever = test_retriever(vector_store, chunks_with_embeddings)
    test_generator(retriever)
    
    # 测试完整流程
    test_rag_pipeline()


if __name__ == "__main__":
    main()
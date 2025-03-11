#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
向量存储测试脚本

这个脚本专门用于测试RAG系统中的向量存储组件。
"""

import os
import sys
import time
import numpy as np
from pathlib import Path

# 添加项目根目录到系统路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.vector_store import SimpleVectorStore


def test_simple_vector_store():
    """测试SimpleVectorStore的基本功能"""
    print("\n===== 测试SimpleVectorStore =====")
    
    # 创建向量存储
    store = SimpleVectorStore()
    print("创建了SimpleVectorStore实例")
    
    # 准备测试数据
    test_documents = [
        {
            'content': '人工智能是计算机科学的一个分支，致力于研究和开发能够模拟人类智能的系统。',
            'metadata': {'source': 'test_doc_1', 'page': 1},
            'embedding': np.random.rand(384)  # 随机生成384维向量
        },
        {
            'content': '机器学习是人工智能的核心，它使计算机能够从数据中学习，而无需明确编程。',
            'metadata': {'source': 'test_doc_2', 'page': 1},
            'embedding': np.random.rand(384)
        },
        {
            'content': '深度学习是机器学习的一个子集，它使用多层神经网络来模拟人脑的工作方式。',
            'metadata': {'source': 'test_doc_3', 'page': 1},
            'embedding': np.random.rand(384)
        },
        {
            'content': '大语言模型是一种基于深度学习的自然语言处理模型，通过在海量文本上训练。',
            'metadata': {'source': 'test_doc_4', 'page': 1},
            'embedding': np.random.rand(384)
        },
        {
            'content': 'RAG系统结合检索系统和生成模型，旨在克服大语言模型的知识局限性。',
            'metadata': {'source': 'test_doc_5', 'page': 1},
            'embedding': np.random.rand(384)
        }
    ]
    
    # 测试添加文档
    print("\n1. 测试添加文档")
    start_time = time.time()
    store.add_documents(test_documents)
    end_time = time.time()
    
    print(f"  - 添加了 {len(test_documents)} 个文档到向量存储")
    print(f"  - 处理时间: {end_time - start_time:.4f} 秒")
    print(f"  - 向量维度: {store.vectors.shape}")
    
    # 测试搜索
    print("\n2. 测试搜索功能")
    # 创建一个查询向量 (与第一个文档相似)
    query_vector = test_documents[0]['embedding'] * 0.9 + np.random.rand(384) * 0.1
    
    start_time = time.time()
    results = store.search(query_vector, top_k=3)
    end_time = time.time()
    
    print(f"  - 搜索时间: {end_time - start_time:.4f} 秒")
    print(f"  - 搜索结果数量: {len(results)}")
    
    for i, result in enumerate(results):
        print(f"    结果 {i+1}: 相似度分数 = {result['score']:.4f}")
        print(f"    内容: {result['content']}")
    
    # 测试保存和加载
    print("\n3. 测试保存和加载功能")
    save_dir = os.path.join(project_root, "examples", "test_vector_store_data")
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存
    start_time = time.time()
    store.save(save_dir)
    end_time = time.time()
    print(f"  - 保存时间: {end_time - start_time:.4f} 秒")
    print(f"  - 向量存储已保存到 {save_dir}")
    
    # 加载
    new_store = SimpleVectorStore()
    start_time = time.time()
    new_store.load(save_dir)
    end_time = time.time()
    
    print(f"  - 加载时间: {end_time - start_time:.4f} 秒")
    print(f"  - 从 {save_dir} 加载了向量存储")
    print(f"  - 加载的文档数量: {len(new_store.documents)}")
    print(f"  - 加载的向量维度: {new_store.vectors.shape}")
    
    # 验证加载后的搜索结果
    print("\n4. 验证加载后的搜索功能")
    start_time = time.time()
    new_results = new_store.search(query_vector, top_k=3)
    end_time = time.time()
    
    print(f"  - 搜索时间: {end_time - start_time:.4f} 秒")
    print(f"  - 搜索结果数量: {len(new_results)}")
    
    for i, result in enumerate(new_results):
        print(f"    结果 {i+1}: 相似度分数 = {result['score']:.4f}")
        print(f"    内容: {result['content']}")
    
    # 验证结果一致性
    is_consistent = all(results[i]['score'] == new_results[i]['score'] for i in range(len(results)))
    print(f"\n结果一致性检查: {'通过' if is_consistent else '失败'}")
    
    return store


def main():
    """主函数"""
    print("开始测试向量存储组件...\n")
    test_simple_vector_store()
    print("\n测试完成!")


if __name__ == "__main__":
    main()
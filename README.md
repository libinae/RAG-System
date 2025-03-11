# RAG系统（检索增强生成）

这是一个从头实现的RAG（Retrieval-Augmented Generation，检索增强生成）系统，用于增强大语言模型的知识检索能力。

## 项目结构

```
├── data/                  # 存放文档数据
├── src/                   # 源代码
│   ├── document_loader.py # 文档加载和预处理
│   ├── text_chunker.py    # 文本分块
│   ├── embeddings.py      # 向量嵌入生成
│   ├── vector_store.py    # 向量数据库
│   ├── retriever.py       # 检索模块
│   ├── generator.py       # 生成模块
│   └── rag_pipeline.py    # RAG流程集成
├── examples/              # 示例代码
├── requirements.txt       # 项目依赖
└── main.py                # 主程序入口
```

## 安装

```bash
pip install -r requirements.txt
```

## 使用方法

1. 准备文档数据，放入`data/`目录
2. 运行主程序：

```bash
python main.py
```

## 功能特点

- 支持多种文档格式（PDF、TXT、DOCX等）的加载和处理
- 灵活的文本分块策略
- 高效的向量嵌入生成
- 可扩展的向量数据库支持
- 相似度检索算法
- 与大语言模型的无缝集成

## 系统流程

1. **文档加载与预处理**：加载各种格式的文档，并进行基础的文本清洗
2. **文本分块**：将长文本分割成适合处理的小块
3. **向量嵌入**：为每个文本块生成向量表示
4. **向量存储**：将向量和对应的文本块存入向量数据库
5. **检索**：根据用户查询，检索最相关的文本块
6. **生成**：将检索到的信息与用户查询一起发送给大语言模型，生成增强的回答
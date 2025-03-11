#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RAG系统主程序

这个脚本提供了一个命令行界面来使用RAG系统的各种功能。
"""

import os
import sys
from typing import Optional

import typer
from rich.console import Console
from rich.prompt import Prompt

from src.rag_pipeline import create_rag_pipeline

# 创建Typer应用和Rich控制台
app = typer.Typer(help="检索增强生成(RAG)系统")
console = Console()


@app.command("index")
def index_documents(
    documents_dir: str = typer.Argument(..., help="文档目录路径"),
    config_path: Optional[str] = typer.Option(None, "--config", "-c", help="配置文件路径"),
    save_dir: Optional[str] = typer.Option(None, "--save", "-s", help="保存RAG系统的目录路径")
):
    """索引文档目录"""
    try:
        # 创建RAG流程
        console.print("[bold green]创建RAG流程...[/bold green]")
        rag = create_rag_pipeline(config_path)
        
        # 索引文档
        console.print(f"[bold green]开始索引文档目录: {documents_dir}[/bold green]")
        rag.index_documents(documents_dir)
        
        # 保存系统状态
        if save_dir:
            console.print(f"[bold green]保存RAG系统到: {save_dir}[/bold green]")
            rag.save(save_dir)
            
        console.print("[bold green]索引完成![/bold green]")
    except Exception as e:
        console.print(f"[bold red]错误: {str(e)}[/bold red]")
        sys.exit(1)


@app.command("query")
def query(
    query_text: Optional[str] = typer.Argument(None, help="查询文本"),
    rag_dir: str = typer.Option(..., "--dir", "-d", help="RAG系统目录路径"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="返回的最相似文档数量")
):
    """处理用户查询"""
    try:
        # 创建并加载RAG流程
        console.print(f"[bold green]加载RAG系统: {rag_dir}[/bold green]")
        rag = create_rag_pipeline()
        rag.load(rag_dir)
        
        # 如果没有提供查询文本，则进入交互模式
        if not query_text:
            console.print("[bold cyan]进入交互查询模式，输入'exit'或'quit'退出[/bold cyan]")
            while True:
                query_text = Prompt.ask("\n[bold yellow]请输入您的问题[/bold yellow]")
                if query_text.lower() in ["exit", "quit"]:
                    break
                    
                # 处理查询
                console.print("[cyan]正在检索和生成回答...[/cyan]")
                answer = rag.query(query_text, top_k=top_k)
                
                # 显示回答
                console.print("\n[bold green]回答:[/bold green]")
                console.print(answer)
        else:
            # 处理单次查询
            console.print("[cyan]正在检索和生成回答...[/cyan]")
            answer = rag.query(query_text, top_k=top_k)
            
            # 显示回答
            console.print("\n[bold green]回答:[/bold green]")
            console.print(answer)
    except Exception as e:
        console.print(f"[bold red]错误: {str(e)}[/bold red]")
        sys.exit(1)


@app.command("interactive")
def interactive(
    rag_dir: str = typer.Option(..., "--dir", "-d", help="RAG系统目录路径"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="返回的最相似文档数量")
):
    """启动交互式查询模式"""
    query(None, rag_dir, top_k)


@app.command("create-config")
def create_config(
    output_path: str = typer.Argument(..., help="输出配置文件路径")
):
    """创建默认配置文件"""
    import json
    
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
            'type': 'simple',  # 可选: 'simple', 'faiss', 'chroma'
            'params': {}
        },
        'retriever': {
            'type': 'vector',  # 可选: 'vector', 'hybrid'
            'params': {
                'top_k': 5
            }
        },
        'generator': {
            'type': 'template',  # 可选: 'template', 'openai'
            'params': {}
        }
    }
    
    # 确保目录存在
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # 写入配置文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(default_config, f, ensure_ascii=False, indent=2)
    
    console.print(f"[bold green]默认配置文件已创建: {output_path}[/bold green]")


if __name__ == "__main__":
    app()
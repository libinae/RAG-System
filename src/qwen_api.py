"""阿里巴巴Qwen API模块

这个模块提供了对阿里巴巴Qwen API的支持，包括生成和嵌入功能。
"""

import os
from typing import List, Dict, Any, Optional, Union

import numpy as np

# 尝试导入DashScope，如果不可用则提供警告
try:
    from dashscope import Generation, MultiModalEmbedding

    import dashscope
    DASHSCOPE_AVAILABLE = True
except ImportError:
    DASHSCOPE_AVAILABLE = False
    print("警告: DashScope未安装，将无法使用QwenGenerator和QwenEmbedder")

# 从base_generator.py导入Generator基类
from .base_generator import Generator

# 从embeddings.py导入TextEmbedder基类
from .embeddings import TextEmbedder


class QwenGenerator(Generator):
    """基于阿里巴巴Qwen API的生成器"""
    
    def __init__(self, model_name: str = "qwen-max", api_key: Optional[str] = None, temperature: float = 0.7):
        """初始化Qwen生成器
        Args:
            model_name: Qwen模型名称，默认为qwen-max
            api_key: DashScope API密钥，如果为None则从环境变量获取
            temperature: 生成温度，控制创造性
        """
        super().__init__()
        
        if not DASHSCOPE_AVAILABLE:
            raise ImportError("DashScope未安装，请先安装dashscope库")
        
        self.model_name = model_name
        self.temperature = temperature
        
        # 直接设置API密钥到dashscope模块
        if api_key:
            dashscope.api_key = api_key
        elif 'DASHSCOPE_API_KEY' in os.environ:
            dashscope.api_key = os.environ['DASHSCOPE_API_KEY']
        else:
            raise ValueError("必须提供DashScope API密钥或设置DASHSCOPE_API_KEY环境变量")
    
    def generate(self, query: str, contexts: List[Dict[str, Any]]) -> str:
        """使用Qwen生成回答
        
        Args:
            query: 用户查询
            contexts: 检索到的相关文档列表
            
        Returns:
            生成的回答
        """
        if not contexts:
            return "抱歉，我没有找到相关的信息来回答您的问题。"
        
        # 构建提示
        prompt = "请根据以下信息回答问题。只使用提供的信息，如果信息不足，请说明无法回答。\n\n"
        
        # 添加检索到的内容
        prompt += "参考信息:\n"
        for i, context in enumerate(contexts, 1):
            content = context.get('content', '')
            prompt += f"[{i}] {content}\n\n"
        
        # 添加用户问题
        prompt += f"问题: {query}\n"
        prompt += "回答:"
        
        # 调用Qwen API
        response = Generation.call(
            model=self.model_name,
            prompt=prompt,
            temperature=self.temperature,
            result_format='message',  # 使用message格式获取结构化输出
        )
        
        if response.status_code == 200:
            return response.output.choices[0].message.content
        else:
            error_msg = f"API调用失败: {response.code}, {response.message}"
            print(error_msg)
            return f"抱歉，生成回答时出现错误: {error_msg}"


class QwenEmbedder(TextEmbedder):
    """基于阿里巴巴Qwen API的文本嵌入器"""
    
    def __init__(self, model_name: str = 'text-embedding-v2', api_key: Optional[str] = None):

        """初始化Qwen嵌入器
        
        Args:
            model_name: Qwen嵌入模型名称
            api_key: DashScope API密钥，如果为None则从环境变量获取
        """
        super().__init__()
        
        if not DASHSCOPE_AVAILABLE:
            raise ImportError("DashScope未安装，请先安装dashscope库")
        
        self.model_name = model_name
        
        # 直接设置API密钥到dashscope模块
        if api_key:
            dashscope.api_key = api_key
        elif 'DASHSCOPE_API_KEY' in os.environ:
            dashscope.api_key = os.environ['DASHSCOPE_API_KEY']
        else:
            raise ValueError("必须提供DashScope API密钥或设置DASHSCOPE_API_KEY环境变量")
    
    def embed_text(self, text: str) -> np.ndarray:
        """将单个文本转换为向量"""
        response = MultiModalEmbedding.call(
            model=self.model_name,
            texts=[text],
        )
        
        if response.status_code == 200:
            return np.array(response.output.embeddings[0].embedding)
        else:
            error_msg = f"API调用失败: {response.code}, {response.message}"
            print(error_msg)
            raise RuntimeError(f"嵌入生成失败: {error_msg}")
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """将多个文本转换为向量"""
        response = MultiModalEmbedding.call(
            model=self.model_name,
            texts=texts,
        )
        
        if response.status_code == 200:
            return np.array([item.embedding for item in response.output.embeddings])
        else:
            error_msg = f"API调用失败: {response.code}, {response.message}"
            print(error_msg)
            raise RuntimeError(f"嵌入生成失败: {error_msg}")
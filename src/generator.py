"""生成模块

这个模块负责将检索到的相关文本与用户查询一起发送给大语言模型，生成增强的回答。
"""

import os
from typing import List, Dict, Any, Optional, Union

# 尝试导入OpenAI，如果不可用则提供警告
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("警告: OpenAI未安装，将无法使用OpenAIGenerator")

# 设置Qwen可用性标志
QWEN_AVAILABLE = True
try:
    import dashscope
except ImportError:
    QWEN_AVAILABLE = False
    print("警告: DashScope未安装，将无法使用QwenGenerator")


# 从base_generator.py导入Generator基类
from .base_generator import Generator


class TemplateGenerator(Generator):
    """基于模板的简单生成器"""
    
    def generate(self, query: str, contexts: List[Dict[str, Any]]) -> str:
        """使用简单模板生成回答
        
        Args:
            query: 用户查询
            contexts: 检索到的相关文档列表
            
        Returns:
            生成的回答
        """
        if not contexts:
            return "抱歉，我没有找到相关的信息来回答您的问题。"
        
        # 构建回答
        answer = f"根据我找到的信息，关于'{query}'的回答是：\n\n"
        
        # 添加检索到的内容
        for i, context in enumerate(contexts, 1):
            content = context.get('content', '')
            source = context.get('metadata', {}).get('source', '未知来源')
            score = context.get('score', 0.0)
            
            answer += f"参考文档 {i} (相关度: {score:.2f}):\n"
            answer += f"来源: {source}\n"
            answer += f"内容: {content[:300]}" + ("..." if len(content) > 300 else "") + "\n\n"
        
        return answer


class OpenAIGenerator(Generator):
    """基于OpenAI API的生成器"""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", api_key: Optional[str] = None, temperature: float = 0.7):
        """初始化OpenAI生成器
        
        Args:
            model_name: OpenAI模型名称
            api_key: OpenAI API密钥，如果为None则从环境变量获取
            temperature: 生成温度，控制创造性
        """
        super().__init__()
        
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI未安装，请先安装openai库")
        
        self.model_name = model_name
        self.temperature = temperature
        
        if api_key:
            self.client = OpenAI(api_key=api_key)
        elif 'OPENAI_API_KEY' in os.environ:
            self.client = OpenAI()
        else:
            raise ValueError("必须提供OpenAI API密钥或设置OPENAI_API_KEY环境变量")
    
    def generate(self, query: str, contexts: List[Dict[str, Any]]) -> str:
        """使用OpenAI生成回答
        
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
        
        # 调用OpenAI API
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "你是一个有用的助手，会根据提供的信息回答问题。"},
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature
        )
        
        return response.choices[0].message.content


class GeneratorFactory:
    """生成器工厂类"""
    
    @staticmethod
    def get_generator(generator_type: str, **kwargs) -> Generator:
        """根据类型返回相应的生成器
        
        Args:
            generator_type: 生成器类型，支持'template'、'openai'和'qwen'
            **kwargs: 传递给生成器的参数
            
        Returns:
            生成器实例
        """
        if generator_type == 'template':
            return TemplateGenerator()
        elif generator_type == 'openai':
            model_name = kwargs.get('model_name', 'gpt-3.5-turbo')
            api_key = kwargs.get('api_key', None)
            temperature = kwargs.get('temperature', 0.7)
            return OpenAIGenerator(model_name=model_name, api_key=api_key, temperature=temperature)
        elif generator_type == 'qwen':
            if not QWEN_AVAILABLE:
                raise ImportError("QwenGenerator未导入成功，请确保已安装dashscope库")
            # 延迟导入QwenGenerator，避免循环导入
            from .qwen_api import QwenGenerator
            model_name = kwargs.get('model_name', 'qwen-max')
            api_key = kwargs.get('api_key', None)
            temperature = kwargs.get('temperature', 0.7)
            return QwenGenerator(model_name=model_name, api_key=api_key, temperature=temperature)
        else:
            raise ValueError(f"不支持的生成器类型: {generator_type}")
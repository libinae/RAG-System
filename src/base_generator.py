"""生成器基类模块
这个模块定义了生成器的基类，用于从检索到的上下文和用户查询生成回答。
"""

from typing import List, Dict, Any

class Generator:
    """生成器基类"""
    
    def __init__(self):
        pass
    
    def generate(self, query: str, contexts: List[Dict[str, Any]]) -> str:
        """根据查询和上下文生成回答"""
        raise NotImplementedError("子类必须实现generate方法")
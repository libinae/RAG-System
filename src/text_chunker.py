"""文本分块模块

这个模块负责将长文本分割成适合处理的小块，以便于后续的向量化和检索。
"""

import re
from typing import List, Dict, Any, Optional, Union


class TextChunker:
    """文本分块器基类"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """初始化文本分块器
        
        Args:
            chunk_size: 每个文本块的最大字符数
            chunk_overlap: 相邻文本块之间的重叠字符数
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def split_text(self, text: str) -> List[str]:
        """将文本分割成块"""
        raise NotImplementedError("子类必须实现split_text方法")


class CharacterTextChunker(TextChunker):
    """基于字符的文本分块器"""
    
    def split_text(self, text: str) -> List[str]:
        """按字符数量分割文本
        
        Args:
            text: 要分割的文本
            
        Returns:
            文本块列表
        """
        if not text:
            return []
            
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            # 计算当前块的结束位置
            end = min(start + self.chunk_size, text_len)
            
            # 如果不是最后一块，尝试在句子或段落边界处分割
            if end < text_len:
                # 尝试在句号、问号、感叹号后分割
                for punct in ['.', '?', '!', '\n\n']:
                    pos = text.rfind(punct, start, end)
                    if pos != -1 and pos + 1 > start + self.chunk_size // 2:
                        end = pos + 1
                        break
            
            # 添加当前块到结果列表
            chunks.append(text[start:end].strip())
            
            # 更新下一块的起始位置，考虑重叠
            start = end - self.chunk_overlap
            
            # 确保起始位置不会后退
            start = max(start, 0)
        
        return chunks


class SentenceTextChunker(TextChunker):
    """基于句子的文本分块器"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        super().__init__(chunk_size, chunk_overlap)
        # 句子分割的正则表达式
        self.sentence_pattern = re.compile(r'(?<=[.!?。！？]\s)')
    
    def split_text(self, text: str) -> List[str]:
        """按句子分割文本
        
        Args:
            text: 要分割的文本
            
        Returns:
            文本块列表
        """
        if not text:
            return []
            
        # 分割成句子
        sentences = self.sentence_pattern.split(text)
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            # 如果当前句子加上已有内容超过了块大小，开始新块
            if current_size + sentence_size > self.chunk_size and current_chunk:
                chunks.append(''.join(current_chunk).strip())
                
                # 保留重叠部分的句子
                overlap_size = 0
                overlap_chunk = []
                
                for s in reversed(current_chunk):
                    if overlap_size + len(s) <= self.chunk_overlap:
                        overlap_chunk.insert(0, s)
                        overlap_size += len(s)
                    else:
                        break
                
                current_chunk = overlap_chunk
                current_size = overlap_size
            
            current_chunk.append(sentence)
            current_size += sentence_size
        
        # 添加最后一个块
        if current_chunk:
            chunks.append(''.join(current_chunk).strip())
        
        return chunks


class ParagraphTextChunker(TextChunker):
    """基于段落的文本分块器"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """初始化段落文本分块器
        
        Args:
            chunk_size: 每个文本块的最大字符数
            chunk_overlap: 相邻文本块之间的重叠字符数
        """
        super().__init__(chunk_size, chunk_overlap)
        # 预先创建字符分块器，避免在split_text中重复创建
        self.char_chunker = CharacterTextChunker(chunk_size, chunk_overlap)
    
    def split_text(self, text: str) -> List[str]:
        """按段落分割文本
        
        Args:
            text: 要分割的文本
            
        Returns:
            文本块列表
        """
        if not text:
            return []
        
        # 使用更高效的段落分割方法
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
        
        # 如果没有段落，返回空列表
        if not paragraphs:
            return []
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for paragraph in paragraphs:
            paragraph_size = len(paragraph)
            
            # 如果当前段落太大，使用字符分块器进一步分割
            if paragraph_size > self.chunk_size:
                # 如果当前块不为空，先保存
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk).strip())
                    current_chunk = []
                    current_size = 0
                
                # 使用字符分块器处理大段落
                paragraph_chunks = self.char_chunker.split_text(paragraph)
                chunks.extend(paragraph_chunks)
                continue
            
            # 如果当前段落加上已有内容超过了块大小，开始新块
            if current_chunk and (current_size + paragraph_size > self.chunk_size):
                chunks.append('\n\n'.join(current_chunk).strip())
                
                # 优化重叠部分计算 - 使用更高效的算法
                if self.chunk_overlap > 0:
                    # 计算需要保留的段落
                    new_current_chunk = []
                    new_size = 0
                    
                    # 从后向前添加段落，直到达到重叠大小
                    for p in reversed(current_chunk):
                        if new_size + len(p) + 2 <= self.chunk_overlap:
                            new_current_chunk.insert(0, p)
                            new_size += len(p) + 2
                        else:
                            break
                    
                    current_chunk = new_current_chunk
                    current_size = new_size - 2 if new_size > 0 else 0  # 减去最后一个段落后的换行符
                else:
                    current_chunk = []
                    current_size = 0
            
            current_chunk.append(paragraph)
            current_size += paragraph_size + 2  # +2 for '\n\n'
        
        # 添加最后一个块
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk).strip())
        
        return chunks


class ChunkerFactory:
    """文本分块器工厂类"""
    
    @staticmethod
    def get_chunker(chunker_type: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> TextChunker:
        """根据类型返回相应的分块器
        
        Args:
            chunker_type: 分块器类型，可选值为 'character', 'fixed_size', 'sentence', 'paragraph'
            chunk_size: 每个文本块的最大字符数
            chunk_overlap: 相邻文本块之间的重叠字符数
            
        Returns:
            文本分块器实例
        """
        if chunker_type in ['character', 'fixed_size']:  # 将fixed_size映射到character分块器
            return CharacterTextChunker(chunk_size, chunk_overlap)
        elif chunker_type == 'sentence':
            return SentenceTextChunker(chunk_size, chunk_overlap)
        elif chunker_type == 'paragraph':
            return ParagraphTextChunker(chunk_size, chunk_overlap)
        else:
            raise ValueError(f"不支持的分块器类型: {chunker_type}")


def split_documents(documents: Dict[str, str], chunker_type: str = 'paragraph', 
                    chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Dict[str, Any]]:
    """分割多个文档
    
    Args:
        documents: 包含文件路径和文本内容的字典
        chunker_type: 分块器类型
        chunk_size: 每个文本块的最大字符数
        chunk_overlap: 相邻文本块之间的重叠字符数
        
    Returns:
        包含文档信息和分块内容的字典列表
    """
    # 创建一个分块器实例，避免为每个文档重新创建
    chunker = ChunkerFactory.get_chunker(chunker_type, chunk_size, chunk_overlap)
    chunks = []
    
    # 预先分配足够大的列表空间，减少动态扩展开销
    estimated_chunks_per_doc = 5  # 估计每个文档的平均块数
    total_estimated_chunks = len(documents) * estimated_chunks_per_doc
    chunks = [None] * total_estimated_chunks
    chunk_index = 0
    
    for doc_path, content in documents.items():
        # 跳过空内容
        if not content.strip():
            continue
            
        doc_chunks = chunker.split_text(content)
        total_doc_chunks = len(doc_chunks)
        
        # 如果预分配空间不足，扩展列表
        if chunk_index + total_doc_chunks > len(chunks):
            chunks.extend([None] * max(total_doc_chunks, len(chunks) // 2))
        
        for i, chunk_text in enumerate(doc_chunks):
            chunks[chunk_index] = {
                'document': doc_path,
                'chunk_id': i,
                'text': chunk_text,
                'metadata': {
                    'source': doc_path,
                    'chunk_index': i,
                    'total_chunks': total_doc_chunks
                }
            }
            chunk_index += 1
    
    # 移除未使用的预分配空间
    return chunks[:chunk_index]
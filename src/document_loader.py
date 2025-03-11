"""文档加载和预处理模块

这个模块负责加载各种格式的文档（PDF、TXT、DOCX等）并进行基础的文本清洗。
"""

import os
import re
from typing import List, Dict, Any, Optional, Union

import PyPDF2
import docx2txt
from PIL import Image
import pytesseract
from tqdm import tqdm


class DocumentLoader:
    """文档加载器基类"""
    
    def __init__(self, allowed_extensions: list = None):
        self.allowed_extensions = allowed_extensions or []

    def validate_file(self, file_path: str) -> None:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件 {file_path} 不存在")
        
        if not os.path.isfile(file_path):
            raise ValueError(f"路径 {file_path} 不是文件")
        
        ext = os.path.splitext(file_path)[1].lower()
        if self.allowed_extensions and ext not in self.allowed_extensions:
            raise ValueError(f"不支持的文件类型 {ext}")

    def load(self, file_path: str) -> str:
        """加载文档并返回文本内容"""
        raise NotImplementedError("子类必须实现load方法")
    
    def clean_text(self, text: str) -> str:
        """清理文本内容"""
        # 移除多余的空白字符
        text = re.sub(r'\s+', ' ', text)
        # 移除特殊字符
        text = re.sub(r'[^\w\s.,;:!?\-\'\"\(\)\[\]{}]', '', text)
        return text.strip()


class TextLoader(DocumentLoader):
    """纯文本文档加载器"""
    
    def load(self, file_path: str) -> str:
        """加载文本文件"""
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        return self.clean_text(text)


class PDFLoader(DocumentLoader):
    """PDF文档加载器"""
    
    def __init__(self):
        super().__init__(allowed_extensions=['.pdf'])

    def load(self, file_path: str) -> str:
        try:
            self.validate_file(file_path)
            text = ''
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    text += page.extract_text() + '\n'
            return self.clean_text(text)
        except (PyPDF2.errors.PdfReadError, ValueError) as e:
            raise RuntimeError(f"加载PDF文件失败: {str(e)}") from e


class DocxLoader(DocumentLoader):
    """Word文档加载器"""
    
    def __init__(self):
        super().__init__(allowed_extensions=['.docx'])

    def load(self, file_path: str) -> str:
        try:
            self.validate_file(file_path)
            return self.clean_text(docx2txt.process(file_path))
        except (ValueError, docx2txt.FileNotFoundError) as e:
            raise RuntimeError(f"加载DOCX文件失败: {str(e)}") from e


class ImageLoader(DocumentLoader):
    """图片文档加载器（OCR）"""
    
    def load(self, file_path: str) -> str:
        """加载图片文件并进行OCR"""
        image = Image.open(file_path)
        text = pytesseract.image_to_string(image, lang='chi_sim+eng')
        return self.clean_text(text)


class DocumentLoaderFactory:
    """文档加载器工厂类"""
    
    @staticmethod
    def get_loader(file_path: str) -> DocumentLoader:
        """根据文件类型返回相应的加载器"""
        _, ext = os.path.splitext(file_path.lower())
        
        if ext == '.txt' or ext == '.md':
            return TextLoader()
        elif ext == '.pdf':
            return PDFLoader()
        elif ext == '.docx' or ext == '.doc':
            return DocxLoader()
        elif ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
            return ImageLoader()
        else:
            raise ValueError(f"不支持的文件类型: {ext}")


def load_documents(directory: str) -> List[Dict[str, Any]]:
    """加载目录中的所有文档
    
    Args:
        directory: 文档目录路径
        
    Returns:
        包含文档内容和元数据的列表
    """
    documents = []
    
    # 遍历目录中的所有文件
    for root, _, files in os.walk(directory):
        for file in tqdm(files, desc="加载文档"):
            file_path = os.path.join(root, file)
            
            try:
                # 获取适合的加载器
                loader = DocumentLoaderFactory.get_loader(file_path)
                
                # 加载文档内容
                content = loader.load(file_path)
                
                # 创建文档对象
                document = {
                    'content': content,
                    'metadata': {
                        'source': file_path,
                        'filename': file,
                        'filetype': os.path.splitext(file)[1][1:]
                    }
                }
                
                documents.append(document)
            except Exception as e:
                print(f"加载文档 {file_path} 时出错: {str(e)}")
    
    return documents
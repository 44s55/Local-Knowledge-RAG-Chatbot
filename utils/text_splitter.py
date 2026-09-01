# utils/text_splitter.py
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.config import settings


class TextSplitter:
    @staticmethod
    def split_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """递归字符文本切分"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "，", " ", ""]
        )
        chunks = splitter.split_text(text)
        return chunks

    @staticmethod
    def split(text: str) -> List[str]:
        """使用全局Settings配置，业务层统一入口，不用传参"""
        return TextSplitter.split_text(
            text,
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )

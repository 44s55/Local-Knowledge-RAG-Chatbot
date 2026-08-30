# utils/embedder.py
from typing import List
import os
import dashscope
from dashscope import TextEmbedding


class Embedder:
    """调用阿里云百炼线上Embedding接口，无需本地下载模型文件
    环境变量：读取 .env 中的 LLM_API_KEY
    """

    def __init__(self):
        # 适配你本地.env：变量名为 LLM_API_KEY
        dashscope.api_key = os.getenv("LLM_API_KEY")
        self.model_name = "text-embedding-v2"
        self.dimension = 1024

    def embed_text(self, text: str) -> List[float]:
        """单条文本向量，用于用户query查询"""
        resp = TextEmbedding.call(model=self.model_name, input=text)
        if resp.status_code == 200:
            return resp.output["embeddings"][0]["embedding"]
        raise RuntimeError(f"向量接口调用失败:{resp.code},{resp.message}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量文档切片向量化，知识库入库"""
        resp = TextEmbedding.call(model=self.model_name, input=texts)
        if resp.status_code == 200:
            emb_list = [item["embedding"] for item in resp.output["embeddings"]]
            return emb_list
        raise RuntimeError(f"批量向量接口调用失败:{resp.code},{resp.message}")

    def embed_chunks(self, chunks: List[dict]) -> List[dict]:
        """直接处理切片后的字典列表，给每一段增加embedding字段"""
        texts = [c["content"] for c in chunks]
        embeddings = self.embed_documents(texts)
        for chunk, embedding in zip(chunks, embeddings):
            chunk["embedding"] = embedding
        return chunks

# utils/embedder.py
from typing import List, Dict
import os
import dashscope
from dashscope import TextEmbedding
from dotenv import load_dotenv

# 加载项目根目录 .env 文件
load_dotenv()


class Embedder:
    """调用阿里云百炼线上Embedding接口，无需本地下载模型文件
    环境变量：.env 配置 LLM_API_KEY=你的百炼key
    """

    def __init__(self):
        self.api_key = os.getenv("LLM_API_KEY")
        if not self.api_key:
            raise ValueError("未读取到环境变量 LLM_API_KEY，请检查 .env 文件")
        dashscope.api_key = self.api_key

        self.model_name = "text-embedding-v2"
        self.dimension = 1024
        # 阿里云embedding接口单次最大批量条数
        self.batch_max_size = 25

    def embed_text(self, text: str) -> List[float]:
        """单条文本向量，用于用户query查询"""
        if not text or not text.strip():
            raise ValueError("输入文本不能为空")

        resp = TextEmbedding.call(model=self.model_name, input=text.strip())
        if resp.status_code == 200:
            return resp.output["embeddings"][0]["embedding"]
        raise RuntimeError(f"向量接口调用失败: {resp.code}, {resp.message}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量文档切片向量化，知识库入库
        自动过滤空字符串，超过单次上限会截断（如需分片循环可以再扩展）
        """
        # 过滤空文本
        valid_texts = [t.strip() for t in texts if t and t.strip()]
        if not valid_texts:
            return []

        # 阿里云接口限制，单次最多25条
        if len(valid_texts) > self.batch_max_size:
            raise RuntimeError(f"批量向量单次最多{self.batch_max_size}条，请拆分切片列表")

        resp = TextEmbedding.call(model=self.model_name, input=valid_texts)
        if resp.status_code == 200:
            emb_list = [item["embedding"] for item in resp.output["embeddings"]]
            return emb_list
        raise RuntimeError(f"批量向量接口调用失败:{resp.code},{resp.message}")

    def embed_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """直接处理切片后的字典列表，给每一段增加embedding字段"""
        # 过滤掉content为空的块
        valid_chunks = [c for c in chunks if c.get("content") and c.get("content").strip()]
        if not valid_chunks:
            raise ValueError("embed_chunks：传入的全部chunk的content都为空，无法生成向量")

        texts = [c["content"] for c in valid_chunks]
        embeddings = self.embed_documents(texts)
        for chunk, embedding in zip(valid_chunks, embeddings):
            chunk["embedding"] = embedding
        return valid_chunks

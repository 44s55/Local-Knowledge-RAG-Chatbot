# utils/vector_store.py
import os
import chromadb
from abc import ABC, abstractmethod
from chromadb.config import Settings
from typing import List, Dict
from utils.embedder import Embedder
os.environ["CHROMA_TELEMETRY_ENABLED"] = "False"

class BaseVectorStore(ABC):
    """向量存储抽象基类，统一接口，方便后续切换 Chroma / FAISS"""

    @abstractmethod
    def add_chunks(self, chunks: List[Dict]) -> None:
        pass

    @abstractmethod
    def search(self, query_text: str, top_k: int = 3) -> List[Dict]:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass


class VectorStore(BaseVectorStore):
    def __init__(self, persist_directory: str, embedder: Embedder):
        """
        :param persist_directory: 向量数据库持久化保存文件夹路径
        :param embedder: 传入我们自己实现的Embedder实例
        """
        self.persist_directory = persist_directory
        self.embedder = embedder

        # 初始化chroma持久化客户端，代码强制关闭遥测，解决版本冲突红色报错
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(
                anonymized_telemetry=False
            )
        )
        # 集合名称
        self.collection_name = "knowledge_base"
        # 向量维度，当前dashscope返回1536
        self.dimension = 1536

        # 获取或者新建集合
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name
        )

    def add_chunks(self, chunks: List[Dict]):
        """
        将切片后的文档块入库
        chunks格式示例：[{"content":"xxx","metadata":{"source":"xxx"}}]
        """
        # 先调用embedder批量生成embedding，拿到处理后的新chunk列表（带有embedding字段）
        processed_chunks = self.embedder.embed_chunks(chunks)

        ids = []
        documents = []
        embeddings = []
        metadatas = []

        for idx, item in enumerate(processed_chunks):
            doc_id = f"doc_{idx}"
            ids.append(doc_id)
            documents.append(item["content"])
            embeddings.append(item["embedding"])
            metadatas.append(item.get("metadata", {}))

        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )
        print(f"[VectorStore.add_chunks] 成功入库 {len(processed_chunks)} 条切片")

    def search(self, query_text: str, top_k: int = 3) -> List[Dict]:
        """根据用户问题检索知识库"""
        query_embedding = self.embedder.embed_text(query_text)
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        output = []
        # 解析检索结果
        for i in range(len(result["documents"][0])):
            output.append({
                "content": result["documents"][0][i],
                "distance": result["distances"][0][i],
                "metadata": result["metadatas"][0][i]
            })
        return output

    def clear(self):
        """清空当前知识库集合"""
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(name=self.collection_name)


class ChromaVectorStore:
    pass

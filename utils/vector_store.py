# utils/vector_store.py
import os
# 关闭Chroma遥测上报，消除控制台冗余报错
os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"

import chromadb
from abc import ABC, abstractmethod
from chromadb.config import Settings
from typing import List, Dict, Any
from utils.embedder import Embedder
from config.logger import logger

# 关闭chroma遥测，消除控制台telemetry报错
os.environ["CHROMA_TELEMETRY_ENABLED"] = "False"


class BaseVectorStore(ABC):
    """向量存储抽象基类，统一接口，方便后续切换 Chroma / FAISS"""

    @abstractmethod
    def add_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        pass

    @abstractmethod
    def search(self, query_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
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

    def add_chunks(self, chunks: List[Dict[str, Any]]):
        """
        将切片后的文档块入库
        chunks格式示例：[{"content":"xxx","metadata":{"source":"xxx"}}]
        """
        if not chunks:
            return

        import hashlib
        batch_size = self.embedder.batch_max_size
        processed_chunks = []
        # 分批embedding，规避单次embedding数量上限
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batch_result = self.embedder.embed_chunks(batch)
            processed_chunks.extend(batch_result)

        ids = []
        documents = []
        embeddings = []
        metadatas = []

        for idx, item in enumerate(processed_chunks):
            # 使用文件名哈希 + 切片序号生成全局唯一ID，避免重复上传ID冲突
            src_name = item.get("metadata", {}).get("source", "unknown")
            hash_digest = hashlib.md5(src_name.encode("utf-8")).hexdigest()
            doc_id = f"{hash_digest}_{idx}"

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

    def search(self, query_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        根据用户问题检索知识库
        返回格式：[{"content":"文本","metadata":dict,"distance":float}, ...]
        保证metadata永远是字典对象，不会返回None，防止 .get() 调用报错
        """
        query_embedding = self.embedder.embed_text(query_text)
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        output = []
        doc_list = result["documents"][0]
        dist_list = result["distances"][0]
        meta_list = result["metadatas"][0]

        for i in range(len(doc_list)):
            raw_meta = meta_list[i]
            # 核心修复：chroma部分版本会返回metadata=None，强制转为空字典
            safe_meta = raw_meta if raw_meta is not None else {}
            output.append({
                "content": doc_list[i],
                "distance": dist_list[i],
                "metadata": safe_meta
            })
        return output

    def clear(self):
        """清空当前知识库集合"""
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(name=self.collection_name)


class ChromaVectorStore:
    pass

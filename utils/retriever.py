# utils/retriever.py
from typing import List, Dict
from utils.embedder import Embedder
from utils.vector_store import VectorStore
from config.settings import settings


class Retriever:
    """语义检索器"""

    def __init__(self, embedder: Embedder = None, vector_store: VectorStore = None):
        self.embedder = embedder or Embedder()
        # 从配置读取向量库路径，必须传入参数，禁止无参实例化
        self.vector_store = vector_store or VectorStore(
            persist_directory=settings.VECTOR_DB_PATH,
            embedder=self.embedder
        )

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        results = self.vector_store.search(
            query_text=query,
            top_k=top_k
        )
        print(f"[retrieve调试] results={results}")
        return results

    def get_context(self, query: str, top_k: int = 5) -> tuple:
        results = self.retrieve(query, top_k=top_k)
        context_parts = []
        sources = []
        for i, r in enumerate(results):
            # 来源存放于metadata字典
            source_name = r["metadata"].get("source", "未知文档")
            context_parts.append(f"[来源{i+1}: {source_name}]\n{r['content']}")
            sources.append({
                "source": source_name,
                "score": r["distance"],
            })
        context = "\n\n".join(context_parts)
        return context, sources

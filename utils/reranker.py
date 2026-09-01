import os
import requests
from utils.config import settings


class Reranker:
    """
    文本重排器：阿里云百炼gte‑rerank，HTTP原生接口，不依赖dashscope内部类
    """
    def __init__(self, use_cloud: bool = True, threshold: float = 0.0):
        self.use_cloud = use_cloud
        self.threshold = threshold
        self.api_key = os.getenv("DASHSCOPE_API_KEY", "")
        self.api_url = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank"

    def rerank(self, query: str, documents: list, top_k: int = None) -> list:
        """
        :param query: 用户问题
        :param documents: 文档列表，元素dict，含有content / page_content, metadata
        :param top_k: 返回数量
        :return: 重排后的文档列表，metadata写入rerank_score
        """
        if not documents or not self.api_key:
            return documents

        doc_texts = [doc.get("page_content", doc.get("content", "")) for doc in documents]
        return_n = top_k if top_k else len(documents)

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content‑Type": "application/json"
            }
            payload = {
                "model": "gte‑rerank",
                "query": query,
                "documents": doc_texts,
                "top_n": return_n
            }
            resp = requests.post(self.api_url, json=payload, headers=headers, timeout=30)
            resp.raise_for_status()
            result = resp.json()

            if result.get("code") and result["code"] != "200":
                raise Exception(f"rerank api error:{result.get('message')}")

            reranked_docs = []
            for item in result["output"]["results"]:
                idx = item["index"]
                score = item["relevance_score"]
                if score < self.threshold:
                    continue
                doc = documents[idx]
                doc["metadata"]["rerank_score"] = round(score, 4)
                reranked_docs.append(doc)
            return reranked_docs

        except Exception as e:
            raise Exception(f"重排调用失败: {str(e)}")

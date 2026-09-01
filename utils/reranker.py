# utils/reranker.py
import os
import requests
from utils.config import settings


class Reranker:
    """
    文本重排器：阿里云百炼 qwen3-rerank，HTTP原生接口
    """
    def __init__(self, use_cloud: bool = True, threshold: float = 0.0):
        self.use_cloud = use_cloud
        self.threshold = threshold
        self.api_key = os.getenv("DASHSCOPE_API_KEY", "")
        self.api_url = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"

    def rerank(self, query: str, documents: list, top_k: int = None) -> list:
        """
        :param query: 用户问题
        :param documents: 文档列表，元素dict，含有content / page_content, metadata
        :param top_k: 返回数量
        :return: 重排后的文档列表，metadata写入rerank_score
        """
        if not documents or not self.api_key:
            if not self.api_key:
                print("[Reranker] 警告：API密钥为空，跳过重排")
            return documents

        # 提取文档文本列表
        doc_texts = [doc.get("page_content", doc.get("content", "")) for doc in documents]
        return_n = top_k if top_k else len(documents)

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            # 修复：正确的请求格式，input包裹query和documents，parameters包裹top_n
            payload = {
                "model": "qwen3-rerank",
                "input": {
                    "query": query,
                    "documents": doc_texts
                },
                "parameters": {
                    "top_n": return_n,
                    "return_documents": False
                }
            }
            resp = requests.post(self.api_url, json=payload, headers=headers, timeout=30)
            resp.raise_for_status()
            result = resp.json()

            # 校验返回码
            if result.get("code") and str(result["code"]) != "200":
                raise Exception(f"接口错误:{result.get('message', '未知错误')}")

            reranked_docs = []
            results = result.get("output", {}).get("results", [])
            if not results:
                print("[Reranker] 接口返回空结果，降级使用原始检索")
                return documents

            for item in results:
                idx = item["index"]
                score = item["relevance_score"]
                # 阈值过滤
                if score < self.threshold:
                    continue
                # 按索引取回原文档，写入重排分数
                doc = documents[idx]
                doc["metadata"]["rerank_score"] = round(score, 4)
                reranked_docs.append(doc)

            return reranked_docs if reranked_docs else documents

        except Exception as e:
            print(f"[Reranker] 重排调用失败，降级使用原始结果: {str(e)}")
            return documents


def get_reranker():
    """
    工厂函数：根据settings配置生成Reranker实例
    """
    if not settings.RERANK_ENABLE:
        print("ℹ️ Reranker总开关关闭，跳过重排初始化")
        return None

    if settings.RERANK_USE_CLOUD:
        instance = Reranker(
            use_cloud=settings.RERANK_USE_CLOUD,
            threshold=settings.RERANK_THRESHOLD
        )
        print(f"✅ 云端Reranker实例创建成功，阈值={settings.RERANK_THRESHOLD}")
        return instance
    else:
        print("ℹ️ 当前配置为本地Reranker，暂未实现")
        return None

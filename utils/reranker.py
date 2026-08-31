from typing import List, Dict, Any
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from sentence_transformers import CrossEncoder
import dashscope
from http import HTTPStatus
from dotenv import load_dotenv

# 定位项目根目录加载环境变量
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"))


class Reranker:
    """
    本地交叉编码器重排 BAAI/bge‑reranker‑base
    ⚠️面试踩坑：国内网络实例化会触发HuggingFace下载超时 WinError 10060
    解决方案：
        1.配置HF镜像；
        2.预先把模型下载到本地磁盘，传入本地路径；
        3.切换DashScope云端重排规避下载问题。
    """
    def __init__(self, model_name: str = "BAAI/bge-reranker-base", device: str = None):
        self.cross_encoder = CrossEncoder(model_name, device=device)

    def rerank(self, query: str, docs: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
        if len(docs) == 0:
            return []

        pairs = []
        for doc in docs:
            pairs.append([query, doc["page_content"]])

        scores = self.cross_encoder.predict(pairs)

        scored_result = []
        for idx, doc in enumerate(docs):
            scored_result.append({
                "page_content": doc["page_content"],
                "metadata": doc["metadata"],
                "rerank_score": float(scores[idx])
            })
        scored_result.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored_result[:top_k]


class DashScopeReranker:
    """
    阿里云百炼云端重排，无需本地模型权重
    注意：重排使用模型 qwen3‑rerank，对话生成使用 qianwen‑turbo，二者不能混用
    """
    def rerank(self, query: str, docs: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
        if len(docs) == 0:
            return []

        # 仅调用时读取密钥，模块导入阶段不初始化dashscope
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError("未读取到DASHSCOPE_API_KEY，请检查.env配置")
        dashscope.api_key = api_key

        text_list = [d["page_content"] for d in docs]
        resp = dashscope.TextReRank.call(
            model="qwen3-rerank",
            query=query,
            documents=text_list,
            top_n=top_k,
            return_documents=False
        )
        scored_results = []
        if resp.status_code == HTTPStatus.OK:
            for res_item in resp.output.results:
                origin_doc = docs[res_item.index]
                scored_results.append({
                    "page_content": origin_doc["page_content"],
                    "metadata": origin_doc["metadata"],
                    "rerank_score": res_item.relevance_score
                })
        else:
            # API调用失败降级策略：不做重排，直接返回原始文档切片，防止RAG完全失效
            print(f"[重排API调用失败] code:{resp.status_code}, msg:{resp.message}")
            return docs[:top_k]
        return scored_results


if __name__ == "__main__":
    """自测重排模块，不发起真实网络请求"""
    test_docs = [
        {"page_content":"RRF用来做多路检索融合","metadata":{"source":"test1"}},
        {"page_content":"大模型会产生幻觉问题","metadata":{"source":"test2"}}
    ]
    print("reranker模块加载完成")

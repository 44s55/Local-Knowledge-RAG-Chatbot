# utils/bm25_retriever.py
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
import jieba


class BM25Retriever:
    """
    BM25关键词检索器
    用于混合检索：弥补向量检索在关键词、专有名词上召回不足的问题
    """
    def __init__(self):
        # 存储原始文档文本
        self.doc_texts: List[str] = []
        # 存储文档对应的元数据
        self.doc_metas: List[Dict[str, Any]] = []
        # BM25索引实例
        self.bm25: BM25Okapi | None = None

    def build_index(self, texts: List[str], metas: List[Dict[str, Any]]):
        """
        构建BM25索引
        :param texts: 文档文本列表
        :param metas: 每篇文档对应的元数据列表，和texts一一对应
        """
        self.doc_texts = texts
        self.doc_metas = metas
        # 使用jieba做中文分词，BM25需要分词后的token列表
        tokenized_corpus = [jieba.lcut(text) for text in texts]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        根据query做BM25检索
        :param query: 用户查询问题
        :param top_k: 返回topN结果
        :return: 检索结果列表，包含content、metadata、bm25_score
        """
        if self.bm25 is None:
            raise ValueError("BM25索引尚未构建，请先调用build_index()")

        # 用户查询分词
        query_tokens = jieba.lcut(query)
        # 获取bm25打分
        bm25_scores = self.bm25.get_scores(query_tokens)
        # 将分数、文本、元数据打包，再按分数降序排序
        scored_docs = []
        for idx, score in enumerate(bm25_scores):
            scored_docs.append({
                "content": self.doc_texts[idx],
                "metadata": self.doc_metas[idx],
                "bm25_score": float(score)
            })
        # 分数越高代表匹配度越高，降序
        scored_docs.sort(key=lambda x: x["bm25_score"], reverse=True)
        return scored_docs[:top_k]


if __name__ == "__main__":
    # ----------------单元测试代码----------------
    # 用来单独验证本模块是否正常工作
    test_texts = [
        "RAG检索增强生成，分为文档加载、切片、向量化、检索、生成",
        "Agent智能体包含记忆、工具、规划、反思四大组件",
        "BM25是传统关键词检索算法，擅长专有名词匹配"
    ]
    test_metas = [{"source": "测试1"}, {"source": "测试2"}, {"source": "测试3"}]

    bm25_ret = BM25Retriever()
    bm25_ret.build_index(test_texts, test_metas)
    res = bm25_ret.search("RAG分为哪些步骤", top_k=2)
    print("====BM25检索测试结果====")
    for item in res:
        print(f"score:{item['bm25_score']:.3f} | content:{item['content']}")

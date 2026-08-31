from typing import List, Dict, Any
import numpy as np
from rank_bm25 import BM25Okapi
from utils.vector_store import BaseVectorStore
from utils.reranker import DashScopeReranker


class HybridRetriever:
    """
    混合检索器：BM25稀疏关键词检索 + 向量稠密检索 + RRF倒数排名融合
    支持开关控制是否开启重排
    """
    def __init__(
        self,
        vector_store: BaseVectorStore,
        top_n_sparse: int = 10,
        top_n_dense: int = 10,
        final_top_k: int = 4,
        enable_rerank: bool = False
    ):
        self.vector_store = vector_store
        self.top_n_sparse = top_n_sparse
        self.top_n_dense = top_n_dense
        self.final_top_k = final_top_k
        self.enable_rerank = enable_rerank

        self.bm25: BM25Okapi | None = None
        self.corpus_docs: List[Dict[str, Any]] = []

    def build_bm25_index(self, docs: List[Dict[str, Any]]) -> None:
        """基于传入文档构建BM25索引"""
        self.corpus_docs = docs
        tokenized_corpus = [doc["content"].split() for doc in docs]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def _bm25_retrieve(self, query: str) -> List[Dict[str, Any]]:
        """BM25稀疏检索"""
        if self.bm25 is None:
            return []
        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)
        top_idx = np.argsort(scores)[::-1][:self.top_n_sparse]
        return [self.corpus_docs[i] for i in top_idx]

    def _dense_retrieve(self, query: str) -> List[Dict[str, Any]]:
        """向量稠密检索"""
        return self.vector_store.search(query, top_k=self.top_n_dense)

    @staticmethod
    def _rrf_fuse(results_list: List[List[Dict[str, Any]]], k: int = 60) -> List[Dict[str, Any]]:
        """
        RRF Reciprocal Rank Fusion 倒数排名融合算法
        :param results_list: 多个检索返回的文档列表
        :param k: RRF超参数，一般取60
        """
        doc_score: Dict[int, float] = {}
        doc_mapping: Dict[int, Dict[str, Any]] = {}

        for result_set in results_list:
            for rank, doc in enumerate(result_set):
                doc_id = id(doc)
                doc_mapping[doc_id] = doc
                rrf_score = 1.0 / (rank + 1 + k)
                if doc_id in doc_score:
                    doc_score[doc_id] += rrf_score
                else:
                    doc_score[doc_id] = rrf_score

        sorted_ids = sorted(doc_score.keys(), key=lambda x: doc_score[x], reverse=True)
        return [doc_mapping[did] for did in sorted_ids]

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """对外统一检索入口"""
        sparse_res = self._bm25_retrieve(query)
        dense_res = self._dense_retrieve(query)

        fused_docs = self._rrf_fuse([sparse_res, dense_res])
        fused_docs = fused_docs[: self.final_top_k * 2]

        if self.enable_rerank and len(fused_docs) > 0:
            # 适配reranker：把content临时映射成page_content，调用完再还原
            for d in fused_docs:
                d["page_content"] = d["content"]
            reranker = DashScopeReranker()
            rerank_out = reranker.rerank(query, fused_docs, top_k=self.final_top_k)
            for d in rerank_out:
                del d["page_content"]
            return rerank_out
        else:
            return fused_docs[: self.final_top_k]


if __name__ == "__main__":
    # 自测：只测试BM25+RRF融合逻辑，不启用重排，不调用API
    mock_docs = [
        {"content": "RAG分为文档加载、切片、向量化、检索、重排、生成", "metadata": {"source": "note1"}},
        {"content": "BM25擅长关键词字面匹配，向量检索擅长语义理解", "metadata": {"source": "note2"}},
        {"content": "RRF倒数排名融合用来合并多路检索结果", "metadata": {"source": "note3"}},
        {"content": "大模型幻觉是生成和知识库无关的虚假内容", "metadata": {"source": "note4"}}
    ]

    # vector_store这里自测不传真实向量库，只跑bm25+rrf部分
    class MockVectorStore(BaseVectorStore):
        def add_chunks(self, chunks): pass
        def clear(self): pass
        def search(self, query_text, top_k):
            return mock_docs[:2]

    retriever = HybridRetriever(
        vector_store=MockVectorStore(),
        enable_rerank=False,
        final_top_k=3
    )
    retriever.build_bm25_index(mock_docs)
    output = retriever.retrieve("什么是RRF融合")

    print("===混合检索输出结果===")
    for idx, item in enumerate(output):
        print(f"{idx+1}. {item['content']} | source:{item['metadata']['source']}")

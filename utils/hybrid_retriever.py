from typing import List, Dict, Any
import numpy as np
from rank_bm25 import BM25Okapi
from utils.vector_store import BaseVectorStore
from utils.reranker import Reranker
from config.settings import settings
from config.logger import logger


class HybridRetriever:
    """
    混合检索器：BM25稀疏关键词检索 + 向量稠密检索 + RRF倒数排名融合
    支持全局开关控制是否开启重排，支持动态传入top_k
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

        # Reranker 初始化，参数完全对齐 config 配置
        self.reranker = Reranker(
            use_cloud=settings.RERANK_USE_CLOUD,
            threshold=settings.RERANK_THRESHOLD
        )

    def retrieve(self, query: str, top_k: int = None, enable_rerank: bool = False):
        """
        【唯一对外检索入口】混合检索全流程
        :param query: 用户查询
        :param top_k: 最终返回片段数量，不传则使用初始化默认值
        :param enable_rerank: 是否开启本次重排，需全局开关同时开启才生效
        :return: 排序后的文档列表
        """
        # 未传参时使用初始化配置的默认值
        if top_k is None:
            top_k = self.final_top_k

        # 1. BM25 稀疏检索
        bm25_docs = self._bm25_retrieve(query, top_k=self.top_n_sparse)
        # 2. 向量稠密检索
        vector_docs = self.vector_store.search(query, top_k=self.top_n_dense)
        # =========【修复】向量检索返回doc，保证顶层一定有 distance、rerank_score =========
        for vec_doc in vector_docs:
            if "distance" not in vec_doc:
                vec_doc["distance"] = None
            if "rerank_score" not in vec_doc:
                vec_doc["rerank_score"] = None

        # 3. RRF 分数融合
        candidates = self._rrf_fusion([bm25_docs, vector_docs])


        # =========新增调试打印========
        if len(candidates) > 0:
            print(f"【DEBUG‑RRF后第一条】{candidates[0]}")
        # =============================

        # 安全兜底：确保候选集永远是列表
        if candidates is None:
            candidates = []



        # 给候选集初始化顶层rerank_score（注意：顶层key，不放进metadata）
        for doc in candidates:
            if "rerank_score" not in doc:
                doc["rerank_score"] = None

        # 4. 重排逻辑：全局开关 + 入参开关同时满足才执行
        if settings.RERANK_ENABLE and enable_rerank:
            try:
                # 临时补全重排模块依赖的字段名
                for doc in candidates:
                    doc["page_content"] = doc["content"]

                reranked_docs = self.reranker.rerank(query, candidates, top_k=top_k)

                # 重排结果兜底
                if reranked_docs is None:
                    print("[Reranker] 警告：重排返回空结果，降级使用原始检索结果")
                    return candidates[:top_k]

                print(f"[Reranker] 重排完成，返回 {len(reranked_docs)} 条片段")
                return reranked_docs

            except Exception as e:
                print(f"[Reranker] 重排调用异常，降级跳过重排: {str(e)}")
                fallback_docs = candidates[:top_k]
                return fallback_docs

        # 未开启重排时直接返回融合结果
        return candidates[:top_k]

    def build_bm25_index(self, docs: List[Dict[str, Any]]) -> None:
        """基于传入文档构建BM25索引"""
        self.corpus_docs = docs
        tokenized_corpus = [doc["content"].split() for doc in docs]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def clear_bm25(self):
        """清空内存中的BM25索引与文档列表"""
        self.bm25 = None
        self.corpus_docs = []
        print("[HybridRetriever] BM25索引已全部清空")

    def _bm25_retrieve(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """BM25稀疏检索，返回标准字典列表，和向量检索格式对齐"""
        if self.bm25 is None:
            return []
        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)
        top_idx = np.argsort(scores)[::-1][:top_k]
        result_list = []
        for i in top_idx:
            doc_item = self.corpus_docs[i]
            result_list.append({
                "content": doc_item["content"],
                "metadata": doc_item["metadata"],
                "distance": 0.0,  # BM25无向量距离，占位
                "rerank_score": None
            })
        return result_list

    @staticmethod
    def _rrf_fusion(results_list: List[List[Dict[str, Any]]], k: int = 60) -> List[Dict[str, Any]]:
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

    def get_context(self, query: str, top_k: int = 5):
        """
        适配RAGChain调用，拼接上下文
        :param query: 用户问题
        :param top_k: 返回文档数量
        :return: 拼接后的上下文字符串, 完整文档字典列表
        """
        doc_list = self.retrieve(query, top_k=top_k)
        context_parts = [doc["content"] for doc in doc_list]
        return "\n".join(context_parts), doc_list

    def add_documents(self, docs: List[Dict[str, Any]]) -> None:
        """
        【对外接口】新增一批文档，写入向量库；
        注意：只写入向量库！不会自动刷新BM25，需要手动调用rebuild_bm25()
        :param docs: [{"content":"xxx", "metadata":{}}]格式的文档块列表
        """
        self.vector_store.add_chunks(docs)

    def rebuild_bm25(self) -> None:
        """
        【对外接口】重建内存BM25索引
        从底层向量库读取全部已存储文档，重新构建BM25Okapi内存索引
        ⚠️因为BM25是内存实例，磁盘向量库新增数据不会自动同步到BM25；
        每次ingest入库完成之后，必须调用这个函数，否则BM25检索不到新入库文档
        """
        coll = self.vector_store.collection
        res = coll.get(include=["documents", "metadatas"])

        doc_texts = res["documents"]
        doc_metas = res["metadatas"]

        all_docs: List[Dict[str, Any]] = []
        for text, meta in zip(doc_texts, doc_metas):
            if text is not None:
                all_docs.append({"content": text, "metadata": meta if meta is not None else {}})

        if len(all_docs) == 0:
            self.bm25 = None
            self.corpus_docs = []
            print("[rebuild_bm25]向量库为空，清空BM25索引")
            return
        self.build_bm25_index(all_docs)
        print(f"[rebuild_bm25]成功重建BM25索引，加载文档数量：{len(self.corpus_docs)}")


if __name__ == "__main__":
    # 自测：只测试BM25+RRF融合逻辑，不启用重排，不调用API
    mock_docs = [
        {"content": "RAG分为文档加载、切片、向量化、检索、重排、生成", "metadata": {"source": "note1"}},
        {"content": "BM25擅长关键词字面匹配，向量检索擅长语义理解", "metadata": {"source": "note2"}},
        {"content": "RRF倒数排名融合用来合并多路检索结果", "metadata": {"source": "note3"}},
        {"content": "大模型幻觉是生成和知识库无关的虚假内容", "metadata": {"source": "note4"}},
    ]

    class MockVectorStore(BaseVectorStore):
        def add_chunks(self, chunks): pass

        def clear(self): pass

        def similarity_search(self, query_text, top_k):
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
        print(f"{idx + 1}. {item['content']} | source:{item['metadata']['source']} | distance:{item['distance']} | rerank_score:{item['rerank_score']}")

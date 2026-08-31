from typing import List, Dict, Any
import numpy as np
import jieba
from rank_bm25 import BM25Okapi

# 导入抽象向量库基类，所有向量存储实现都要继承这个基类，统一search接口
from utils.vector_store import BaseVectorStore
# 导入通义千问重排器，用于多路检索之后做结果精排，提升相关性
from utils.reranker import DashScopeReranker


class HybridRetriever:
    """
    混合检索器：BM25稀疏关键词检索 + 向量稠密检索 + RRF倒数排名融合
    1. BM25：基于词频，抓字面关键词，解决向量检索丢专有名词、编号问题
    2. Dense向量检索：基于语义 embedding，解决同义词、转述语义匹配
    3. RRF倒数排名融合：不需要关心两路分数量级不一致，只靠文档排序位置做加权合并
    4. 可开关重排模块：融合之后再调用重排模型进一步筛选最相关文档
    """

    def __init__(
            self,
            vector_store: BaseVectorStore,  # 向量存储实例，必须实现search方法
            top_n_sparse: int = 10,        # BM25关键词检索，每一路取出多少候选文档
            top_n_dense: int = 10,         # 向量稠密检索，每一路取出多少候选文档
            final_top_k: int = 4,          # 最终返回给大模型的文档数量
            enable_rerank: bool = False    # 是否开启重排，开启会调用API产生token消耗
    ):
        self.vector_store = vector_store
        self.top_n_sparse = top_n_sparse
        self.top_n_dense = top_n_dense
        self.final_top_k = final_top_k
        self.enable_rerank = enable_rerank

        self.bm25: BM25Okapi | None = None  # BM25实例，调用build_bm25_index之后才初始化
        self.corpus_docs: List[Dict[str, Any]] = []  # 原始全量文档，和bm25的token语料一一对应

    def build_bm25_index(self, docs: List[Dict[str, Any]]) -> None:
        """
        基于传入文档构建BM25稀疏检索索引
        注意：
            1. 该方法只需要在文档入库完成之后执行一次，不要每一次query都重建索引
            2. rank_bm25无中文分词能力，使用jieba做中文词语切分
        :param docs: LangChain格式文档列表，每个字典必须包含 page_content 字段存储文本
        """
        # 保存原始文档，后续bm25拿到下标可以直接取出完整文档对象
        self.corpus_docs = docs
        # 使用jieba对中文文档分词，生成词语序列给BM25使用
        tokenized_corpus = [jieba.lcut(doc["page_content"]) for doc in docs]
        # 初始化BM25模型，完成词频、逆文档频率统计，构建检索索引
        self.bm25 = BM25Okapi(tokenized_corpus)

    def _bm25_retrieve(self, query: str) -> List[Dict[str, Any]]:
        """
        BM25稀疏关键词检索（私有内部方法，外部不要直接调用）
        适用场景：用户提问包含专有名词、设备编号、特定术语，向量容易漏召回，BM25可以兜底
        :param query: 用户原始提问字符串
        :return: 根据关键词得分排序的文档列表，最多返回 top_n_sparse条
        """
        # 防御：还没构建bm25索引的时候直接返回空列表，避免报错
        if self.bm25 is None:
            return []

        # 用户query必须和文档使用同一套jieba分词，分词粒度保持一致
        tokenized_query = jieba.lcut(query)
        # bm25.get_scores：输出每一篇文档相对于当前query的bm25匹配分数
        scores = self.bm25.get_scores(tokenized_query)

        # np.argsort：得到分数从小到大的下标；[::-1]反转变成从大到小；切片取前top_n_sparse个下标
        top_idx = np.argsort(scores)[::-1][:self.top_n_sparse]
        # 根据下标从原始文档池取出对应文档，作为bm25检索结果
        return [self.corpus_docs[i] for i in top_idx]

    def _dense_retrieve(self, query: str) -> List[Dict[str, Any]]:
        """
        向量稠密语义检索（私有内部方法）
        原理：query向量化之后，和向量库里面的文档embedding做余弦相似度匹配
        :param query: 用户原始提问字符串
        :return: 语义相似度最高的 top_n_dense 篇文档
        """
        # 调用向量存储统一search接口，底层可以是Chroma、FAISS、Milvus任意实现
        return self.vector_store.search(query, k=self.top_n_dense)

    @staticmethod
    def _rrf_fuse(results_list: List[List[Dict[str, Any]]], k: int = 60) -> List[Dict[str, Any]]:
        """
        RRF Reciprocal Rank Fusion 倒数排名融合算法【静态工具方法】
        核心优势：
            BM25分数范围、向量相似度分数范围完全不一样，不能直接相加；
            RRF不去看原始分数，只看文档在各路结果中的排位，做加权求和，多路检索融合首选。
        公式：rrf_score = Σ( 1 / (rank + 1 + k) )
            rank：文档在当前检索结果里面的位置，从0开始
            k：超参数，业界常用60；k越大排位靠后的文档衰减越慢
        :param results_list: 多路检索结果，例如 [bm25结果列表, 向量检索结果列表]
        :param k: RRF超参数，默认60
        :return: 融合之后整体排序完成的文档列表，不做截断
        """
        doc_score: Dict[int, float] = {}    # key: 对象内存id，value:累计RRF得分
        doc_mapping: Dict[int, Dict[str, Any]] = {}  # 内存id映射回原始文档对象

        for result_set in results_list:
            # rank 是当前文档在这一路检索里面的序号，从0开始，0代表第一名
            for rank, doc in enumerate(result_set):
                # 使用id()取对象内存地址作为唯一标识；注意：不同内存对象内容一样会被识别成两篇
                doc_id = id(doc)
                doc_mapping[doc_id] = doc
                # 套用RRF计算公式，计算该文档在当前这条检索流的贡献分
                rrf_score = 1.0 / (rank + 1 + k)

                # 累加多路得分，如果文档同时出现在BM25和向量结果，分数叠加，排名会更高
                if doc_id in doc_score:
                    doc_score[doc_id] += rrf_score
                else:
                    doc_score[doc_id] = rrf_score

        # 根据累加后的RRF分数降序排序，分数越高越靠前
        sorted_ids = sorted(doc_score.keys(), key=lambda x: doc_score[x], reverse=True)
        # 通过映射表还原文档对象输出
        return [doc_mapping[did] for did in sorted_ids]

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """
        对外统一检索入口，业务层只调用这一个方法
        完整链路：
            1. BM25稀疏检索拿到候选
            2. 向量稠密检索拿到候选
            3. RRF融合两路结果
            4. 预取2倍final_top_k文档留给重排做筛选
            5. 如果开启重排，调用DashScope重排模型；没有开启就直接截断返回
        :param query: 用户的问题
        :return: 最终给到LLM上下文的文档列表，长度等于final_top_k
        """
        # 分别执行两路检索
        sparse_res = self._bm25_retrieve(query)
        dense_res = self._dense_retrieve(query)

        # RRF融合BM25 + 向量两路结果
        fused_docs = self._rrf_fuse([sparse_res, dense_res])

        # 先取2倍数量，给重排留有候选池；不直接取final_top_k，防止重排之后优质文档被提前截断丢掉
        fused_docs = fused_docs[: self.final_top_k * 2]

        # 判断开关：开启重排并且候选文档不为空，则调用重排API做二次精排
        if self.enable_rerank and len(fused_docs) > 0:
            reranker = DashScopeReranker()
            return reranker.rerank(query, fused_docs, top_k=self.final_top_k)
        else:
            # 不开启重排，直接截断返回最终条数
            return fused_docs[: self.final_top_k]


if __name__ == "__main__":
    """
    模块自测入口
    目的：单元测试混合检索逻辑，不依赖真实向量库、不调用任何外部API
    模拟少量测试文档，验证BM25召回、RRF融合逻辑是否正常运行
    """
    # 模拟知识库文档，格式和langchain Document转字典保持一致
    mock_docs = [
        {"page_content": "RAG分为文档加载、切片、向量化、检索、重排、生成", "metadata": {"source": "note1"}},
        {"page_content": "BM25擅长关键词字面匹配，向量检索擅长语义理解", "metadata": {"source": "note2"}},
        {"page_content": "RRF倒数排名融合用来合并多路检索结果", "metadata": {"source": "note3"}},
        {"page_content": "大模型幻觉是生成和知识库无关的虚假内容", "metadata": {"source": "note4"}}
    ]

    # Mock模拟向量存储类，替代真实Chroma/Milvus，仅用于本地自测
    class MockVectorStore:
        def search(self, query, k):
            """模拟向量库search接口，直接返回前两条文档，不需要做真实embedding计算"""
            return mock_docs[:2]

    # 实例化混合检索器，使用mock向量库，关闭重排避免消耗api key
    retriever = HybridRetriever(
        vector_store=MockVectorStore(),
        enable_rerank=False,
        final_top_k=3
    )

    # 构建BM25索引，测试环境传入模拟文档集合
    retriever.build_bm25_index(mock_docs)
    # 模拟用户提问，执行完整retrieve流程
    output = retriever.retrieve("什么是RRF融合")

    # 控制台打印检索输出，肉眼校验召回结果是否符合预期
    print("===混合检索输出结果===")
    for idx, item in enumerate(output):
        print(f"{idx+1}. {item['page_content']} | source:{item['metadata']['source']}")

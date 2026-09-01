# utils/rag_chain.py
from utils.hybrid_retriever import HybridRetriever
from utils.vector_store import BaseVectorStore, VectorStore
from utils.embedder import Embedder
from run_llm import chat_with_llm

RAG_PROMPT_TEMPLATE = """你是知识库问答助手，请严格使用【参考上下文】回答用户问题。
如果参考上下文里面没有答案，如实回答“知识库未查询到相关信息”，禁止编造不存在的内容。

【参考上下文】
{context}

用户问题：{query}
回答：
"""


class RAGChain:
    def __init__(self):
        # 初始化嵌入模型
        self.embedder = Embedder()
        # 使用真实实现类 VectorStore，传入必须的两个参数
        self.vector_store = VectorStore(
            persist_directory="./chroma_db",
            embedder=self.embedder
        )
        self.hybrid_retriever = HybridRetriever(
            vector_store=self.vector_store,
            enable_rerank=False
        )

    def invoke(self, user_query: str, top_k: int = 5):
        """
        完整RAG执行链路
        :param user_query: 用户提问
        :param top_k: 检索返回片段数量
        :return: answer(模型回答), source_list(完整文档字典列表)
        """
        # 1.调用混合检索器，拿到上下文和完整文档列表
        context, source_list = self.hybrid_retriever.get_context(user_query, top_k=top_k)

        # ==========幻觉阈值判断==========
        from utils.config import settings
        vector_docs = [d for d in source_list if isinstance(d, dict) and "distance" in d]
        if len(vector_docs) > 0:
            all_too_far = all(doc["distance"] > settings.RAG_DISTANCE_THRESHOLD for doc in vector_docs)
            if all_too_far:
                return "知识库未查询到相关内容，无法回答该问题。", source_list
        # ================================

        # 兜底：检索不到任何内容
        if not context.strip():
            return "知识库未查询到相关信息", source_list

        # 2.填充prompt模板
        prompt = RAG_PROMPT_TEMPLATE.format(context=context, query=user_query)
        messages = [{"role": "user", "content": prompt}]
        # 3.调用大模型
        answer = chat_with_llm(messages, temperature=0.3)

        return answer, source_list



# =========全局单例：整个项目共用同一个RAGChain实例========
rag_chain_instance = RAGChain()

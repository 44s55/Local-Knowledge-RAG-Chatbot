# utils/rag_chain.py
from openai import OpenAI
from utils.config import settings
from utils.vector_store import VectorStore
from utils.hybrid_retriever import HybridRetriever
from utils.embedder import Embedder



class RAGChain:
    """
    RAG主链路：检索 → 重排（可选） → 上下文组装 → Prompt构建 → 大模型生成
    对外统一 invoke 接口，与web层完全兼容
    """
    def __init__(self):
        # 初始化嵌入模型 + 向量存储
        self.embedder = Embedder()
        self.vector_store = VectorStore(
            persist_directory=settings.VECTOR_DB_PATH,
            embedder=self.embedder
        )

        # 初始化混合检索器（关闭内部自动重排，统一由本类根据界面开关控制）
        self.hybrid_retriever = HybridRetriever(
            vector_store=self.vector_store,
            top_n_sparse=settings.TOP_K_BM25,
            top_n_dense=settings.TOP_K_VECTOR,
            final_top_k=settings.RERANK_TOP_N,
            enable_rerank=False
        )
        # 初始化大模型客户端（对接阿里云百炼OpenAI兼容模式）
        try:
            self.llm_client = OpenAI(
                api_key=settings.LLM_API_KEY,
                base_url=settings.LLM_BASE_URL
            )
            self.llm_available = True
            print(f"✅ RAGChain：大模型客户端初始化成功，模型={settings.LLM_MODEL_NAME}")
        except Exception as e:
            self.llm_available = False
            print(f"⚠️ RAGChain：大模型客户端初始化失败: {str(e)}，将降级返回原始检索片段")

        # RAG专属系统提示：强约束仅基于上下文回答，抑制幻觉
        self.system_prompt = """你是专业的知识库问答助手。
请严格参考下方的【参考上下文】回答用户问题，禁止编造上下文之外的任何信息。
如果参考上下文没有相关答案，请直接回答"根据现有知识库内容，无法回答该问题"。
回答要求：准确、简洁、条理清晰，优先使用原文表述。"""

    def _build_context_text(self, doc_list):
        """将检索到的文档片段拼接成标准上下文字符串"""
        context_parts = []
        for idx, doc in enumerate(doc_list):
            content = doc.get("page_content", doc.get("content", ""))
            context_parts.append(f"【参考片段{idx+1}】\n{content}")
        return "\n\n".join(context_parts)

    def invoke(self, user_query: str, top_k: int = None, enable_rerank: bool = False):
        """
        【对外主接口】执行完整RAG流程
        :param user_query: 用户问题
        :param top_k: 检索返回片段数量
        :param enable_rerank: 是否开启重排（界面复选框传入，动态生效）
        :return: (回答文本, 检索源文档列表)
        """
        # 1. 执行基础混合检索（BM25+向量+RRF融合，不执行内部重排）
        sources = self.hybrid_retriever.retrieve(user_query, top_k=top_k)
        print(f"[RAGChain] 基础检索完成，返回{len(sources)}条候选片段")

        # 2. 根据界面开关动态控制是否执行重排
        if enable_rerank and hasattr(self.hybrid_retriever, 'reranker'):
            try:
                sources = self.hybrid_retriever.reranker.rerank(
                    query=user_query,
                    documents=sources,
                    top_k=top_k if top_k else settings.RERANK_TOP_N
                )
                print(f"[RAGChain] 重排完成，最终返回{len(sources)}条片段")
            except Exception as e:
                print(f"[RAGChain] 重排调用失败，降级使用原始检索结果: {str(e)}")
                # 降级时补全rerank_score字段，避免前端显示异常
                for doc in sources:
                    doc["metadata"]["rerank_score"] = None
        else:
            # 未开启重排，补全字段
            for doc in sources:
                doc["metadata"]["rerank_score"] = None

        # 检索为空兜底
        if not sources:
            return "知识库中没有找到相关内容，请尝试其他问题。", []

        # 3. 大模型可用则生成回答，不可用则直接返回拼接的片段内容
        if self.llm_available:
            # 组装检索上下文
            context = self._build_context_text(sources)
            # 构建对话消息
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"【参考上下文】\n{context}\n\n【用户问题】\n{user_query}"}
            ]

            # 调用大模型生成回答
            try:
                response = self.llm_client.chat.completions.create(
                    model=settings.LLM_MODEL_NAME,
                    messages=messages,
                    temperature=0.1,  # RAG场景低温度，保证准确性
                    stream=False
                )
                answer = response.choices[0].message.content.strip()
                print("[RAGChain] 大模型生成回答完成")
            except Exception as e:
                print(f"[RAGChain] 大模型调用失败，降级返回原始片段: {str(e)}")
                # 降级：直接返回拼接的检索内容
                answer = self._build_context_text(sources)
        else:
            # 大模型不可用，直接返回原始片段
            answer = self._build_context_text(sources)

        return answer, sources

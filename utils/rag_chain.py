from openai import OpenAI
from config.settings import settings
from utils.vector_store import VectorStore
from utils.hybrid_retriever import HybridRetriever
from utils.embedder import Embedder
from config.logger import logger


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

        # 初始化混合检索器
        self.hybrid_retriever = HybridRetriever(
            vector_store=self.vector_store,
            top_n_sparse=settings.TOP_K_BM25,
            top_n_dense=settings.TOP_K_VECTOR,
            final_top_k=settings.RERANK_TOP_N,
            enable_rerank=False
        )

        # 初始化大模型客户端
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

        # RAG专属系统提示
        self.system_prompt = """你是专业的知识库问答助手。
请严格参考下方的【参考上下文】回答用户问题，禁止编造上下文之外的任何信息。
如果参考上下文没有相关答案，请直接回答"根据现有知识库内容，无法回答该问题"。
回答要求：准确、简洁、条理清晰，优先使用原文表述。"""

        # 幻觉拦截阈值（余弦距离，值越小越相似）
        self.hallucination_threshold = 1.2

    def stream_chat(self, query: str, context_docs: list, conversation_history: list = None):
        """
        流式生成回答，逐token返回
        :param query: 用户问题
        :param context_docs: 检索到的参考文档列表
        :param conversation_history: 对话历史
        :return: 生成器，逐块返回文本片段
        """
        # 拼接上下文提示词
        context_str = "\n\n".join([
            f"[文档{i + 1} 来源：{doc.get('source', '未知')}]\n{doc.get('content', '')}"
            for i, doc in enumerate(context_docs)
        ])

        history_prompt = ""
        if conversation_history:
            history_lines = []
            for msg in conversation_history[-3:]:  # 保留最近3轮
                role = "用户" if msg.get("role") == "user" else "助手"
                history_lines.append(f"{role}：{msg.get('content', '')}")
            history_prompt = "\n对话历史：\n" + "\n".join(history_lines)

        system_prompt = f"""你是一个专业的知识库问答助手。
请严格根据下面提供的参考文档回答用户问题，禁止编造参考文档中不存在的信息。
如果参考文档中没有相关内容，请直接回答“知识库中未查询到相关内容”。

参考文档：
{context_str}
{history_prompt}
"""

        # 调用大模型流式接口
        try:
            response = self.llm_client.chat.completions.create(
                model=settings.LLM_MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                stream=True,
                temperature=0.1
            )

            # 逐块返回文本
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"流式大模型调用失败: {str(e)}", exc_info=True)
            yield "\n[系统错误：流式生成失败]"

    def _build_context_text(self, doc_list):
        """将检索到的文档片段拼接成标准上下文字符串"""
        context_parts = []
        for idx, doc in enumerate(doc_list):
            content = doc.get("page_content", doc.get("content", ""))
            context_parts.append(f"【参考片段{idx + 1}】\n{content}")
        return "\n\n".join(context_parts)

    def invoke(self, user_query: str, top_k: int = 4, enable_rerank: bool = False,
                conversation_history: list = None):
        """
        RAG完整执行入口
        :param user_query: 当前用户问题
        :param top_k: 检索返回片段数
        :param enable_rerank: 是否开启重排
        :param conversation_history: 历史对话列表
        :return: (回答文本, 溯源片段列表)
        """
        if conversation_history is None:
            conversation_history = []

        docs = []

        # 1. 检索知识库片段
        docs = self.hybrid_retriever.retrieve(user_query, top_k=top_k)

        # 调试打印距离（验证完可注释）
        print("[调试] 检索文档距离：", [d.get("distance") for d in docs])

        # 2. 幻觉拦截：distance 在文档根层级
        if not docs or all(
                doc.get("distance", 99) > self.hallucination_threshold for doc in docs):
            return "抱歉，知识库中未查询到与您问题相关的内容，请换个问题或补充文档后再试。", []

        # 3. 拼接上下文
        context_str = "\n------\n".join([doc.get("content", doc.get("page_content", "")) for doc in docs])

        # 4. 构造历史对话文本
        history_str = ""
        for msg in conversation_history:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                history_str += f"用户：{content}\n"
            elif role == "assistant":
                history_str += f"助手：{content}\n"

        # 5. 构造完整消息队列
        system_prompt = f"""
你是一个专业的知识库问答助手，请严格根据下面提供的参考资料回答用户问题。
如果参考资料中没有答案，请明确回答不知道，禁止编造内容。
回答简洁准确，重点突出。

参考资料：
{context_str}
"""

        messages = [
            {"role": "system", "content": system_prompt}
        ]
        if history_str.strip():
            messages.append({"role": "user", "content": f"之前的对话：\n{history_str}\n请结合上文回答当前问题。"})
        messages.append({"role": "user", "content": user_query})

        # 6. 调用大模型生成
        try:
            resp = self.llm_client.chat.completions.create(
                model=settings.LLM_MODEL_NAME,
                messages=messages,
                temperature=0.2,
                stream=False
            )
            answer = resp.choices[0].message.content.strip()
        except Exception as e:
            answer = f"回答生成失败：{str(e)}"
            docs = []

        return answer, docs

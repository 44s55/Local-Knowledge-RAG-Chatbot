# utils/rag_chain.py
from utils.retriever import Retriever
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
        self.retriever = Retriever()

    def invoke(self, user_query: str, top_k: int = 5):
        """
        完整RAG执行链路
        :param user_query: 用户提问
        :param top_k: 检索返回片段数量
        :return: answer(模型回答), source_list(引用来源)
        """
        # 1.检索知识库，拿到上下文和来源信息
        context, source_list = self.retriever.get_context(user_query, top_k=top_k)

        # 兜底：检索不到任何内容
        if not context.strip():
            return "知识库未查询到相关信息", source_list

        # 2.填充prompt模板
        prompt = RAG_PROMPT_TEMPLATE.format(context=context, query=user_query)

        messages = [
            {"role": "user", "content": prompt}
        ]
        # 3.调用大模型
        answer = chat_with_llm(messages, temperature=0.3)

        return answer, source_list

import json
from openai import OpenAI
from utils.config import settings
from utils.rag_chain import RAGChain


class SimpleLightAgent:
    """
    轻量化智能Agent（本科入门项目版）
    核心逻辑：纯手写Prompt驱动意图识别 -> 选择工具 -> 执行工具 -> 返回结果
    不引入LangChain等第三方Agent框架，原生实现"思考-决策-执行"闭环
    工具集：
    1. knowledge_search：调用本地RAG知识库检索问答
    2. no_tool：不调用知识库，大模型直接回答（闲聊、简单计算等）
    """
    def __init__(self, rag_chain: RAGChain):
        # 复用已初始化好的RAG链路，不重复创建实例
        self.rag_chain = rag_chain
        # 大模型客户端，复用全局配置
        self.client = OpenAI(
            api_key=settings.LLM_API_KEY,
            base_url=settings.LLM_BASE_URL
        )

        # Agent系统提示词：严格约束输出格式，只做二分类决策
        self.agent_system_prompt = """
你是一个简易智能调度Agent，请分析用户的提问，选择最合适的工具。
可用工具只有两个：
【knowledge_search】用户询问专业知识点、文档内容、资料查询、与上传知识库相关的问题，使用该工具。
【no_tool】用户闲聊、问候、简单常识、简单数学计算、与上传文档无关的问题，使用该工具，直接回答，不检索知识库。

输出必须严格是JSON格式，不要任何额外文字、不要markdown、不要解释：
{"tool":"工具名","thought":"简短的决策原因"}
"""

    def _plan(self, user_query: str) -> dict:
        """
        Agent思考决策环节：调用大模型判断意图，输出工具选择
        """
        try:
            resp = self.client.chat.completions.create(
                model=settings.LLM_MODEL_NAME,
                messages=[
                    {"role": "system", "content": self.agent_system_prompt},
                    {"role": "user", "content": user_query}
                ],
                temperature=0,  # 意图识别用0温度，保证稳定
                stream=False
            )
            content = resp.choices[0].message.content.strip()
            decision = json.loads(content)
            return decision
        except Exception as e:
            print(f"[Agent] 决策解析失败，降级默认知识库检索: {str(e)}")
            # 异常兜底：默认走知识库检索，保证服务可用
            return {"tool": "knowledge_search", "thought": "决策异常，默认使用知识库检索"}

    def run(self, user_query: str, top_k: int = 4, enable_rerank: bool = False, conversation_history: list = None):
        """
        Agent对外主入口：完整执行一次问答
        :param user_query: 用户问题
        :param top_k: 检索片段数量
        :param enable_rerank: 是否开启重排
        :param conversation_history: 历史对话列表
        :return: (回答文本, 溯源片段列表, 决策信息字典)
        """
        # Step1：意图决策
        decision = self._plan(user_query)
        tool = decision.get("tool", "knowledge_search")
        thought = decision.get("thought", "")
        print(f"[Agent] 决策：{thought} | 选择工具：{tool}")

        sources = []
        if tool == "knowledge_search":
            # Step2-A：调用RAG，传入历史对话
            answer, sources = self.rag_chain.invoke(
                user_query=user_query,
                top_k=top_k,
                enable_rerank=enable_rerank,
                conversation_history=conversation_history
            )
        elif tool == "no_tool":
            # Step2-B：闲聊也传入历史，支持连续闲聊
            try:
                messages = []
                if conversation_history:
                    messages.extend(conversation_history)
                messages.append({"role": "user", "content": user_query})
                messages.insert(0, {"role": "system", "content": "请简洁、友好地回答用户问题"})

                resp = self.client.chat.completions.create(
                    model=settings.LLM_MODEL_NAME,
                    messages=messages,
                    temperature=0.3,
                    stream=False
                )
                answer = resp.choices[0].message.content.strip()
            except Exception as e:
                answer = f"回答生成失败：{str(e)}"
        else:
            answer = "无法识别指令，请针对上传文档提问或进行简单交流。"

        return answer, sources, decision

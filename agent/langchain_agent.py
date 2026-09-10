from typing import List, Dict, Any, Callable
from langchain_openai import ChatOpenAI
from langchain_core.tools import Tool
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# 导入已实现的工具
from tool.calculator_tool import calculate_expression, CALCULATOR_TOOL_DEF
from tool.datetime_tool import datetime_tool, DATETIME_TOOL_DEF
from tool.rag_tool import rag_search_tool, RAG_TOOL_DEF


class LangChainToolAgent:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        model: str = "qwen-turbo",
        rag_search_func: Callable = None
    ):
        """
        LangChain 工具调用 Agent
        :param api_key: 大模型 API Key
        :param base_url: API 兼容地址，默认 DashScope OpenAI 兼容模式
        :param model: 模型名称
        :param rag_search_func: 你原有项目的 RAG 检索入口函数
        """
        self.llm = ChatOpenAI(
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=0.1
        )
        self.rag_search_func = rag_search_func
        self.tools = self._register_tools()
        self.agent_executor = self._build_agent()

    def _register_tools(self) -> List[Tool]:
        """注册所有可用工具"""
        tools = [
            Tool(
                name=CALCULATOR_TOOL_DEF["name"],
                description=CALCULATOR_TOOL_DEF["description"],
                func=lambda expr: calculate_expression(expr)["content"]
            ),
            Tool(
                name=DATETIME_TOOL_DEF["name"],
                description=DATETIME_TOOL_DEF["description"],
                func=lambda _: datetime_tool()["content"]  # 修复：占位接收参数
            )
        ]

        # 如果传入了 RAG 检索函数，注册知识库检索工具
        if self.rag_search_func:
            tools.append(
                Tool(
                    name=RAG_TOOL_DEF["name"],
                    description=RAG_TOOL_DEF["description"],
                    func=lambda query: rag_search_tool(query, self.rag_search_func)["content"]
                )
            )
        return tools

    def _build_agent(self) -> AgentExecutor:
        """构建 Agent 执行器"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个智能助手，可以调用工具解决问题。如果问题涉及计算、时间查询、知识库专业内容，请调用对应工具；如果不需要工具，直接回答即可。"),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])

        agent = create_openai_tools_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )

        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=False,
            handle_parsing_errors=True
        )

    def chat(self, query: str, chat_history: List[Dict] = None) -> str:
        """
        对外对话接口
        :param query: 用户问题
        :param chat_history: 对话历史，格式 [{"role": "user", "content": "xxx"}, {"role": "assistant", "content": "xxx"}]
        :return: 最终回答内容
        """
        # 转换历史消息格式
        history_msgs = []
        if chat_history:
            for msg in chat_history:
                if msg["role"] == "user":
                    history_msgs.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    history_msgs.append(AIMessage(content=msg["content"]))

        result = self.agent_executor.invoke({
            "input": query,
            "chat_history": history_msgs
        })
        return result["output"]

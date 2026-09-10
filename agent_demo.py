import sys
import os

# 把项目根目录加入模块搜索路径，确保能找到 config、utils、agent
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from agent.langchain_agent import LangChainToolAgent
from utils.rag_chain import RAGChain
from config.settings import settings

# ========== 1. 初始化 RAG 主链路 ==========
rag_chain = RAGChain()

# ========== 2. 包装成 Agent 工具兼容的函数 ==========
def rag_search_wrapper(query: str) -> str:
    """
    包装 RAGChain.invoke，只返回回答文本，适配 Agent 工具接口
    """
    answer, docs = rag_chain.invoke(
        user_query=query,
        top_k=4,
        enable_rerank=False,
        enable_query_rewrite=False
    )
    return answer

# ========== 3. 配置与初始化 Agent，复用settings读取.env ==========
agent = LangChainToolAgent(
    api_key=settings.LLM_API_KEY,
    base_url=settings.LLM_BASE_URL,
    model=settings.LLM_MODEL_NAME,
    rag_search_func=rag_search_wrapper
)

# ========== 4. 多场景测试 ==========
if __name__ == "__main__":
    print("=== 测试1：数学计算工具 ===")
    print(agent.chat("计算 (128 + 256) * 3 等于多少"))
    print("\n" + "="*50 + "\n")

    print("=== 测试2：时间查询工具 ===")
    print(agent.chat("现在几点了，今天周几"))
    print("\n" + "="*50 + "\n")

    print("=== 测试3：知识库检索（自动调用 RAG） ===")
    print(agent.chat("本系统采用了什么检索方案？"))
    print("\n" + "="*50 + "\n")

    print("=== 测试4：多任务混合（计算+知识库） ===")
    print(agent.chat("RAG的top_k默认是多少，同时计算 99 * 99 等于多少"))
    print("\n" + "="*50 + "\n")

    print("✅ Agent + RAG 全链路测试完成")

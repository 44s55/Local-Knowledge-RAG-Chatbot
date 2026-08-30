"""
RAG 智能问答系统 - 程序入口
企业级工程入口：app.py
职责：组装模块，提供命令行交互入口
"""
import sys
import logging
# 放在最顶部！屏蔽chromadb posthog遥测版本冲突报错，这是补丁代码
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

sys.path.append(".")

from utils.retriever import Retriever
from run_llm import chat_with_llm

# 全局只初始化1次检索器，不要每次问答重复新建对象
retriever_global = Retriever()


def rag_chat(question: str) -> dict:
    """
    RAG主业务函数
    :param question: 用户问题
    :return: dict{question, answer, sources}
    """
    context, source_list = retriever_global.get_context(question, top_k=4)
    print(f"\n[DEBUG调试] source_list = {source_list}")

    system_prompt = """你是基于本地知识库的问答助手。
只允许使用下面【知识库上下文】里面提供的信息作答，严禁使用你自身的内部知识。
如果知识库没有对应信息，直接回复：“知识库未收录该问题相关信息”，禁止编造、禁止补充知识库以外内容。

【知识库上下文】
{}
""".format(context)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]
    answer = chat_with_llm(messages, temperature=0.3)

    return {
        "question": question,
        "answer": answer,
        "sources": source_list
    }


# ====================== 企业标准入口块，所有执行逻辑放这里面 ======================
if __name__ == "__main__":
    print("=====RAG本地知识库问答系统=====")
    while True:
        user_input = input("\n请输入你的问题(输入exit退出): ")
        if user_input.strip().lower() == "exit":
            print("程序退出")
            break
        result = rag_chat(user_input)
        print("\n【AI回答】")
        print(result["answer"])
        print("\n【引用来源】")
        for s in result["sources"]:
            print(f"- {s['source']}，距离：{s['score']:.4f}")

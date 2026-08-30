# run_retriever.py
from utils.retriever import Retriever

if __name__ == "__main__":
    retriever = Retriever()
    ctx, src_list = retriever.get_context("什么是RAG", top_k=2)
    print("====检索得到上下文====")
    print(ctx)
    print("\n====来源信息====")
    for s in src_list:
        print(s)

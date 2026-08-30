# run_rag_chain.py
from utils.rag_chain import RAGChain

if __name__ == "__main__":
    rag = RAGChain()
    question = "在这里替换成你的知识库问题"
    ans, sources = rag.invoke(question)

    print("====模型回答====")
    print(ans)
    print("\n====引用来源====")
    for s in sources:
        print(f"文档:{s['source']}  距离分数:{s['score']:.4f}")

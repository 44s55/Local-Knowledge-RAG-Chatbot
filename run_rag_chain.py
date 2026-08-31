from utils.rag_chain import RAGChain

if __name__ == "__main__":
    rag = RAGChain()
    print("RAG问答系统启动，输入 q 退出")
    while True:
        question = input("\n请输入你的问题：")
        if question.strip().lower() == "q":
            break
        ans, sources = rag.invoke(question)

        print("\n====模型回答====")
        print(ans)
        print("\n====引用来源====")
        for s in sources:
            print(f"文档:{s['source']}  分数:{s['score']:.4f}")

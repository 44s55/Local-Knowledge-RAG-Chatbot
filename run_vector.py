# run_vector.py
from dotenv import load_dotenv
load_dotenv()

from utils.embedder import Embedder
from utils.vector_store import VectorStore

if __name__ == "__main__":
    embedder = Embedder()
    db_path = "./vector_db"
    vs = VectorStore(persist_directory=db_path, embedder=embedder)

    # 测试样例片段
    test_chunks = [
        {
            "content": "RAG检索增强生成，把外部知识库检索结果交给大模型做回答。",
            "embedding": embedder.embed_text("RAG检索增强生成，把外部知识库检索结果交给大模型做回答。"),
            "metadata": {"source": "test.txt"}
        }
    ]

    vs.clear()
    vs.add_chunks(test_chunks)
    res = vs.search("什么是RAG", top_k=2)
    print("检索结果：")
    for item in res:
        print(f"content:{item['content']}, 距离:{item['distance']:.4f}")

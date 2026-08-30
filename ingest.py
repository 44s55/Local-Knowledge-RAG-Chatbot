import sys
sys.path.append(".")

# 下面保留原本所有import不动
from utils.config import settings
from utils.embedder import Embedder
from utils.vector_store import VectorStore
from utils.text_splitter import TextSplitter

def main():
    embedder = Embedder()
    vs = VectorStore(persist_directory=settings.VECTOR_DB_PATH, embedder=embedder)

    # 直接代码内写测试文本，不需要读取data文件夹的文件
    raw_text = """
RAG检索增强生成技术，通过检索外部知识库，把参考上下文交给大模型，用来减少大模型幻觉，提升回答的真实性。
RAG分为：文档加载、文本切片、向量化存储、向量检索、prompt组装、大模型生成这几个步骤。
"""

    # 文本切分
    chunks = TextSplitter.split_text(
        raw_text,
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP
    )

    # 组装入库
    chunk_list = []
    for idx, text in enumerate(chunks):
        chunk_list.append({
            "content": text,
            "embedding": embedder.embed_text(text),
            "metadata": {"source": "内置测试文本"}
        })

    vs.add_chunks(chunk_list)
    print(f"✅成功入库 {len(chunk_list)} 个文本块")


if __name__ == "__main__":
    main()

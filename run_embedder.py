# run_embedder.py
from dotenv import load_dotenv
# 入口第一行！优先加载环境变量，再导入业务模块
load_dotenv()

from utils.embedder import Embedder

if __name__ == "__main__":
    embedder = Embedder()
    vec = embedder.embed_text("RAG技术结合了检索和生成")
    print(f"向量维度: {len(vec)}")
    print(f"向量前5维: {vec[:5]}")

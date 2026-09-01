from typing import List, Dict, Any
from pathlib import Path
from utils.document_loader import DocumentLoader
from utils.text_splitter import TextSplitter
from utils.embedder import Embedder
from utils.vector_store import VectorStore
from utils.hybrid_retriever import HybridRetriever
from config.settings import settings

# 根据脚本位置自动推导项目根目录，git兼容，不受pycharm工作目录影响
SCRIPT_FILE = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_FILE.parent.parent
DATA_FOLDER = PROJECT_ROOT / "data"


class IngestPipeline:
    def __init__(self):
        # 1. 文档加载器：统一处理 txt/md/pdf 格式解析
        self.loader = DocumentLoader()
        # 2. 嵌入模型：和检索端使用完全一致的嵌入器，保证向量空间对齐
        self.embedder = Embedder()
        # 3. 向量存储：和RAGChain同一路径，读写同一个知识库
        self.vector_store = VectorStore(
            persist_directory=settings.VECTOR_DB_PATH,
            embedder=self.embedder
        )
        # 4. 混合检索器：用于文档入库 + 重建BM25稀疏索引
        self.retriever = HybridRetriever(
            vector_store=self.vector_store,
            top_n_sparse=settings.TOP_K_BM25,
            top_n_dense=settings.TOP_K_VECTOR,
            final_top_k=settings.RERANK_TOP_N,
            enable_rerank=False
        )

    def load_file(self, file_path: str) -> str:
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        # 使用DocumentLoader统一入口，自动适配 txt/md/pdf
        return self.loader.load_file(str(p))

    def process_file(self, file_path: str) -> None:
        raw_text = self.load_file(file_path)

        # 空文档直接跳过，不执行切片入库
        if not raw_text or len(raw_text.strip()) == 0:
            print(f"[process_file] ⚠️ {Path(file_path).name} 内容为空，跳过")
            return

        chunks = TextSplitter.split(raw_text)
        docs: List[Dict[str, Any]] = []
        for chunk in chunks:
            docs.append({
                "content": chunk,
                "metadata": {"source": Path(file_path).name}
            })
        self.retriever.add_documents(docs)
        print(f"[process_file] {Path(file_path).name} 切片完成，共{len(docs)}块")

    def process_dir(self, dir_path: str) -> None:
        p = Path(dir_path)
        print(f"正在扫描目录：{p.resolve()}")
        # 支持 txt / pdf / md 三种后缀
        support_suffix = (".txt", ".pdf", ".md")
        for f in p.glob("**/*"):
            if f.suffix.lower() in support_suffix:
                print(f"✅识别到文档: {f.name}")
                self.process_file(str(f))
        self.retriever.rebuild_bm25()
        print("[process_dir] 全部文档处理完毕")


if __name__ == "__main__":
    pipeline = IngestPipeline()
    pipeline.process_dir(str(DATA_FOLDER))

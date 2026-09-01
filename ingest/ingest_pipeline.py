from typing import List, Dict, Any
from pathlib import Path
from utils.document_loader import DocumentLoader
from utils.text_splitter import TextSplitter

# 根据脚本位置自动推导项目根目录，git兼容，不受pycharm工作目录影响
SCRIPT_FILE = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_FILE.parent.parent
DATA_FOLDER = PROJECT_ROOT / "data"


class IngestPipeline:
    def __init__(self):
        # 内部导入，规避循环导入
        from utils.rag_chain import rag_chain_instance
        self.loader = DocumentLoader()
        self.retriever = rag_chain_instance.hybrid_retriever

    def load_file(self, file_path: str) -> str:
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        suffix = p.suffix.lower()
        if suffix == ".pdf":
            return self.loader.load_pdf(str(p))
        else:
            return self.loader.load_txt(str(p))

    def process_file(self, file_path: str) -> None:
        raw_text = self.load_file(file_path)
        # =========新增调试打印=========
        print(f"====原始读取文本长度：{len(raw_text)}====")
        print(f"raw_text[:200] --> {repr(raw_text[:200])}")
        # =============================
        chunks = TextSplitter.split(raw_text)

        # =========新增打印切片结果=========
        print(f"切片得到块数量：{len(chunks)}")
        for idx, c in enumerate(chunks):
            print(f"chunk[{idx}] len={len(c)} repr={repr(c)}")
        # =================================

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
        print(f"正在扫描目录：{p.resolve()}")  # 打印真实绝对路径
        all_files = list(p.glob("**/*"))
        print(f"目录下全部文件数量：{len(all_files)}")
        for f in p.glob("**/*"):
            if f.suffix.lower() in (".txt", ".pdf"):
                print(f"✅识别到文档: {f.name}")
                self.process_file(str(f))
        self.retriever.rebuild_bm25()
        print("[process_dir] 全部文档处理完毕")


if __name__ == "__main__":
    pipeline = IngestPipeline()
    # 使用自动算出的项目根目录下data，不再写死"./data"
    pipeline.process_dir(str(DATA_FOLDER))

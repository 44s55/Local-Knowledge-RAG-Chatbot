from typing import List, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils.config import settings


class TextSplitter:
    """文本智能切分器"""

    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
        separators: Optional[List[str]] = None,
    ):
        # 优先读取全局配置，不传参就用settings
        if chunk_size is None:
            chunk_size = settings.CHUNK_SIZE
        if chunk_overlap is None:
            chunk_overlap = settings.CHUNK_OVERLAP

        if separators is None:
            separators = [
                "\n\n", "\n", "。", ".", "！", "!", "？", "?",
                "；", ";", "，", ",", " ", "",
            ]
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=len,
        )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        if not text or not text.strip():
            return []
        return self.splitter.split_text(text)

    def split_documents(self, documents: List[dict]) -> List[dict]:
        """
        :param documents: [{"content":"原文","source":"文件名"}]
        :return: [{"content":"切片","source":"源文件","chunk_id":序号,"chunk_size":长度}]
        """
        all_chunks = []
        for doc in documents:
            chunks = self.split_text(doc["content"])
            for i, chunk in enumerate(chunks):
                all_chunks.append({
                    "content": chunk,
                    "source": doc.get("source", "unknown"),
                    "chunk_id": i,
                    "chunk_size": len(chunk),
                })
        return all_chunks

    def get_stats(self, chunks: List[dict]) -> dict:
        if not chunks:
            return {"total_chunks": 0}
        sizes = [c["chunk_size"] for c in chunks]
        return {
            "total_chunks": len(chunks),
            "avg_size": sum(sizes) / len(sizes),
            "min_size": min(sizes),
            "max_size": max(sizes),
            "chunk_size_setting": self.chunk_size,
            "overlap_setting": self.chunk_overlap,
        }


if __name__ == "__main__":
    # 项目自测入口：对接document_loader
    from loaders.document_loader import load_documents_from_dir

    # 1.读取data文件夹文档，输出 [(文件名,原始文本)]
    raw_docs = load_documents_from_dir("./data")
    # 转换格式，适配TextSplitter入参
    input_docs = [{"content": text, "source": name} for name, text in raw_docs]

    splitter = TextSplitter()
    chunk_list = splitter.split_documents(input_docs)

    print(f"原始文档数量：{len(input_docs)}")
    stat = splitter.get_stats(chunk_list)
    print(f"切片统计: {stat}")

    for chunk in chunk_list[:3]:
        print(f"\n[块{chunk['chunk_id']}] source:{chunk['source']}")
        print(chunk["content"][:200])

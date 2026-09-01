"""
文档加载器 DocumentLoader
负责读取PDF、TXT、Markdown，提取原始文本字符串
"""
from pathlib import Path

# pdf解析库 PyPDF2
try:
    from PyPDF2 import PdfReader
except ImportError:
    PdfReader = None


class DocumentLoader:
    def load_pdf(self, file_path: str) -> str:
        """读取PDF，返回原始文本"""
        if PdfReader is None:
            raise RuntimeError("未安装PyPDF2，请执行 pip install PyPDF2")
        fp = Path(file_path)
        reader = PdfReader(str(fp))
        full_text = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                full_text.append(page_text)
        return "\n".join(full_text)

    def load_txt(self, file_path: str) -> str:
        """读取txt/markdown纯文本"""
        fp = Path(file_path)
        # 优先utf-8，失败回退gbk
        try:
            text = fp.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = fp.read_text(encoding="gbk", errors="ignore")
        return text

    def load_file(self, file_path: str) -> str:
        """统一入口：根据文件后缀自动分发加载方法，支持 .txt .md .pdf"""
        fp = Path(file_path)
        suffix = fp.suffix.lower()
        if suffix in (".txt", ".md"):
            return self.load_txt(file_path)
        elif suffix == ".pdf":
            return self.load_pdf(file_path)
        else:
            raise ValueError(f"不支持的文件类型 {suffix}，仅支持txt/md/pdf")

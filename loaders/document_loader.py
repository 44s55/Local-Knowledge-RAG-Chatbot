import os
from pathlib import Path
from typing import Optional

# pdf解析
from pypdf import PdfReader
# docx解析
from docx import Document


def load_single_document(file_path: str) -> Optional[str]:
    """
    加载单个文档，根据文件后缀自动选择解析器
    :param file_path: 文件完整路径
    :return :解析完成的文本字符串，读取失败返回None
    """
    path = Path(file_path)
    if not path.exists():
        print(f"文件不存在：{file_path}")
        return None

    suffix = path.suffix.lower()
    content = ""
    try:
        if suffix in (".txt", ".md"):
            # 文本、markdown
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
        elif suffix == ".pdf":
            reader = PdfReader(path)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    content += page_text + "\n"
        elif suffix == ".docx":
            doc = Document(path)
            for para in doc.paragraphs:
                content += para.text + "\n"
        else:
            print(f"不支持的文件格式 {suffix}，跳过 {file_path}")
            return None

    except Exception as e:
        print(f"解析文件异常 {file_path}，错误：{str(e)}")
        return None

    return content.strip()


def load_documents_from_dir(dir_path: str) -> list[tuple[str, str]]:
    """
    遍历整个文件夹，批量加载全部支持的文档
    :param dir_path: 文档目录路径
    :return: list[(文件名, 文件文本内容)]
    """
    result = []
    dir_p = Path(dir_path)
    if not dir_p.is_dir():
        print(f"目录不存在：{dir_path}")
        return result

    support_suffix = {".txt", ".md", ".pdf", ".docx"}
    for file in os.listdir(dir_p):
        f_path = dir_p / file
        if not f_path.is_file():
            continue
        if f_path.suffix.lower() not in support_suffix:
            continue

        text = load_single_document(str(f_path))
        if text:
            result.append((file, text))
    return result


if __name__ == "__main__":
    # 本地测试入口：测试读取data文件夹下面全部文档
    from config.settings import settings
    data_dir = "./data"
    docs = load_documents_from_dir(data_dir)
    print(f"一共读取到 {len(docs)} 个文档")
    for name, txt in docs[:2]:
        print(f"\n=====文档名：{name}=====")
        print(txt[:300])

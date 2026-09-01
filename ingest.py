"""
文档入库流水线 Ingest Pipeline
功能：批量把 PDF / TXT / Markdown 文档解析、切片、向量化存入Chroma向量库，同步构建BM25检索索引

业务流程完整链路：
输入(单文件/文件夹)
    → 遍历扫描文件，过滤允许的后缀
    → 根据文件后缀调用对应文档加载器 DocumentLoader
    → 原始文本清洗：去除多余换行、空白行
    → 文本切片 TextSplitter，生成多个文本块chunk
    → 为每一个文本块绑定元数据metadata（文件名、绝对路径、文件类型，用于网页溯源展示）
    → 将文本块写入Chroma向量持久化库
    → ⚠️重点：新增文档之后，手动重建内存BM25索引，保障混合检索HybridRetriever可以检索到新文档
    → 输出完整处理统计日志，打印失败文件的异常信息

注意点：
1. BM25当前为内存版本：程序重启之后BM25索引消失，需要重新rebuild_bm25()，面试记为待优化点
2. 本模块只做上层组装，DocumentLoader、TextSplitter、HybridRetriever全部复用已写好的utils底层类，不修改底层源码
3. 对外提供统一入口方法 ingest(input_path)，支持传入【单个文件路径】或者【文件夹路径】
4. 后续可以直接给Gradio网页上传接口调用这个入口，网页上传文件之后自动完成入库，不用手动运行脚本
"""

import os
from pathlib import Path  # Path库：跨平台路径处理，自动处理Windows正反斜杠，避免路径bug
from typing import List, Dict, Any  # 类型注解，方便阅读代码，IDE自动提示

# 导入项目已经写好的底层工具类，完全复用，不修改底层
from utils.document_loader import DocumentLoader
from utils.text_splitter import TextSplitter
from utils.rag_chain import rag_chain  # 获取全局实例，里面持有 hybrid_retriever 对象


class IngestPipeline:
    # 定义允许处理的文件后缀，不在此列表的文件直接跳过，不做解析
    SUFFIX_ALLOW = {".pdf", ".txt", ".md"}

    def __init__(self):
        """
        流水线初始化
        实例化我们已经写好的加载器、切分器；拿到全局混合检索器实例
        stat 字典：统计处理状态，最后打印日志输出给开发者看
        """
        self.loader = DocumentLoader()       # 文档加载器：读取pdf/txt原始文本
        self.splitter = TextSplitter()       # 文本切分器：长文本切为多个chunk片段
        self.retriever = rag_chain.hybrid_retriever  # 混合检索器：包含chroma向量库 + 内存BM25

        # 统计信息，记录整个入库任务的运行情况
        self.stat = {
            "total_files": 0,       # 本次一共扫描到多少个待处理文件
            "success_files": 0,     # 处理成功的文件数量
            "fail_files": 0,        # 处理失败的文件数量
            "total_chunks": 0,      # 最终入库成功的文本块总数量
            "fail_list": []         # 存储失败的文件名 + 报错信息，方便排查问题
        }

    def _clean_text(self, text: str) -> str:
        """
        【私有工具方法】简单文本清洗
        问题来源：PDF解析经常出现大量空行、连续换行、大量空格，会严重影响切片质量
        处理逻辑：按行分割，过滤掉空行，再用换行符拼接回文本
        :param text: loader读取出来的原始文档文本
        :return: 清洗之后干净的文本字符串
        """
        # 防御判断，如果传入空字符串，直接返回空，避免后续报错
        if not text:
            return ""

        # 遍历每一行，strip去掉每行首尾空格，过滤掉空字符串
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        # 将过滤后的有效行，重新拼接
        clean_result = "\n".join(lines)
        return clean_result

    def _load_single_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        【私有工具方法】处理单个文件：加载、清洗、切片、绑定元数据
        :param file_path: 文件的绝对/相对路径字符串
        :return: 返回列表，列表内每一项是一个chunk字典 {content:"文本内容", metadata:{元数据字典}}
        """
        # Path包装路径，获取文件名、后缀、绝对路径
        file_obj = Path(file_path)
        suffix = file_obj.suffix.lower()   # 获取文件后缀，全部转小写，兼容 .PDF / .Pdf 大小写混乱情况
        file_name = file_obj.name         # 获取文件名，例如 "杠杆原理.pdf"，溯源的时候展示给前端
        file_abs_path = str(file_obj.resolve())  # 获取文件完整绝对磁盘路径

        raw_text = ""
        # 根据不同后缀，分发调用不同加载器
        if suffix == ".pdf":
            # pdf文件，调用pdf加载器读取文本
            raw_text = self.loader.load_pdf(file_path)
        elif suffix in (".txt", ".md"):
            # txt 和 markdown文件，统一走文本加载器；markdown本质就是纯文本，不需要特殊解析
            raw_text = self.loader.load_txt(file_path)
        else:
            # 不在允许后缀列表，抛出异常，上层捕获计入失败列表
            raise ValueError(f"不支持该文件格式：{suffix}")

        # 执行文本清洗，去除多余换行空格
        clean_text = self._clean_text(raw_text)

        # 极端情况：PDF解析之后全部是图片，没有文字，清洗后文本为空，直接返回空列表，不生成chunk
        if not clean_text:
            return []

        # 调用文本切分器，把长文档切成多个短文本块
        chunk_text_list = self.splitter.split_text(clean_text)

        doc_chunk_list = []
        for chunk_content in chunk_text_list:
            # 给每一个文本块绑定元数据，后续问答溯源读取这里的字段
            meta_data = {
                "source": file_name,          # 溯源核心：文档文件名，网页展示用
                "file_path": file_abs_path,   # 文件磁盘完整路径，调试排查用
                "file_type": suffix.lstrip(".") # 文件类型 pdf / txt / md
            }
            # 组装成字典格式，和retriever add_documents入参格式对齐
            doc_chunk_list.append({
                "content": chunk_content,
                "metadata": meta_data
            })

        return doc_chunk_list

    def ingest(self, input_path: str):
        """
        ========== 对外公开统一入口函数 ==========
        外部脚本、Gradio网页后端，全部调用这个方法启动入库流水线
        :param input_path: 可以传入【单个文件路径】 或者 【文件夹目录路径】
        :return: 返回stat统计字典，调用方可以拿到处理结果，用于网页提示
        """
        input_path_obj = Path(input_path)

        # 防御：判断传入路径是否真实存在，不存在直接抛异常
        if not input_path_obj.exists():
            raise FileNotFoundError(f"传入的路径不存在：{input_path}")

        file_to_process = []
        # 分支1：传入的是一个文件
        if input_path_obj.is_file():
            file_to_process.append(input_path_obj)

        # 分支2：传入的是文件夹，递归扫描整个目录，筛选符合后缀的文档
        elif input_path_obj.is_dir():
            # rglob递归遍历目录下所有子文件
            for each_file in input_path_obj.rglob("*"):
                # 判断：是文件，并且后缀在允许列表
                if each_file.is_file() and each_file.suffix.lower() in self.SUFFIX_ALLOW:
                    file_to_process.append(each_file)

        # 更新统计：待处理总文件数量
        self.stat["total_files"] = len(file_to_process)
        print(f"\n[Ingest流水线启动] 扫描完毕，待处理文件总数: {len(file_to_process)}")

        all_final_chunks = []  # 存储本次所有文件生成的全部chunk，后续统一批量入库

        # 循环逐个处理每一个文件
        for fp in file_to_process:
            fp_str = str(fp)
            try:
                # 调用单文件处理函数，得到该文件的全部chunk
                chunk_result = self._load_single_file(fp_str)

                if len(chunk_result) == 0:
                    print(f"[WARN] {fp.name} 解析完成，但没有提取到有效文本，跳过入库")
                    self.stat["success_files"] += 1
                    continue

                # 将当前文件的chunk，加入总列表
                all_final_chunks.extend(chunk_result)
                self.stat["success_files"] += 1
                print(f"[OK] {fp.name} → 生成 {len(chunk_result)} 个文本块")

            except Exception as err:
                # 捕获单个文件异常，一个文件损坏不影响其他文件继续处理
                self.stat["fail_files"] += 1
                self.stat["fail_list"].append((fp.name, repr(err)))
                print(f"[ERROR] 文件 {fp.name} 处理失败，错误信息：{err}")

        # 更新总chunk统计
        self.stat["total_chunks"] = len(all_final_chunks)

        # 如果本次没有产出任何文本块，直接结束流水线，不执行入库
        if len(all_final_chunks) == 0:
            print("\n[Ingest流水线结束] 没有得到任何可入库的文本块")
            return self.stat

        # --------------------------
        # 步骤1：批量写入Chroma向量持久化库
        # --------------------------
        # add_documents接收格式：[{"content":"xxx","metadata":{}}]，和我们组装的结构完全匹配
        self.retriever.add_documents(all_final_chunks)

        # --------------------------
        # ⚠️超级重点！面试高频坑
        # 向Chroma写入新数据之后，必须手动重建BM25内存索引
        # BM25是内存实例，不会自动感知磁盘chroma新增的数据；不rebuild，混合检索BM25部分拿不到新文档
        # --------------------------
        self.retriever.rebuild_bm25()

        # ============输出完整统计报告============
        print("\n" + "="*70)
        print(f"✅[Ingest流水线执行完成 统计报告]")
        print(f"扫描总文件数：{self.stat['total_files']}")
        print(f"成功处理文件：{self.stat['success_files']}")
        print(f"失败文件数量：{self.stat['fail_files']}")
        print(f"本次总共入库文本块：{self.stat['total_chunks']}")
        if len(self.stat["fail_list"]) > 0:
            print("\n❌处理失败的文件清单：")
            for filename, error_msg in self.stat["fail_list"]:
                print(f"   · {filename}  :  {error_msg}")
        print("="*70 + "\n")

        return self.stat


# ---------------- 本地调试测试入口 ----------------
# 直接运行这个py文件，就会执行，批量处理 ./data目录下面全部文档
# 使用方式：python ingest/ingest_pipeline.py
if __name__ == "__main__":
    print("=====本地测试Ingest流水线，处理./data目录=====")
    pipeline = IngestPipeline()
    # 传入文件夹路径，批量导入data下所有pdf/txt/md
    result_stat = pipeline.ingest(input_path="./data")
    # result_stat就是统计字典，调用方可以拿到数据，网页端就可以把这个信息展示给用户

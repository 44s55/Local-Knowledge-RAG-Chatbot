"""
上下文压缩器模块
功能：对检索返回的文档片段进行相关性压缩与精炼
- 去除检索片段中的冗余、无关、重复内容
- 只保留与用户问题强相关的核心句子与关键信息
- 减少送入大模型的Token数量，提升回答精准度
设计原则：输入输出格式与原检索结果完全兼容，可插拔替换
"""

from openai import OpenAI
from config.settings import settings
from config.logger import logger
from typing import List, Dict
import json


class ContextCompressor:
    """上下文压缩器：基于问题对检索片段做相关性精炼"""

    def __init__(self):
        self.client = OpenAI(
            api_key=settings.LLM_API_KEY,
            base_url=settings.LLM_BASE_URL
        )
        self.model = settings.LLM_MODEL_NAME

    def compress(self, query: str, documents: List[Dict], keep_rate: float = 0.6) -> List[Dict]:
        """
        对检索结果列表进行压缩
        :param query: 用户原始问题
        :param documents: 检索结果列表，每项包含 content、source、score 等字段
        :param keep_rate: 保留比例（0-1），控制压缩程度
        :return: 压缩后的文档列表，字段结构与输入完全一致
        """
        if not documents:
            return []

        # 构造待压缩文本清单
        doc_texts = []
        for i, doc in enumerate(documents):
            doc_texts.append(f"[{i+1}] {doc.get('content', '')}")

        docs_block = "\n\n".join(doc_texts)

        system_prompt = """你是一个专业的文档上下文压缩助手。
任务：针对用户的问题，从给出的多个文档片段中，只保留与问题高度相关的核心内容。
压缩规则：
1. 严格保留与问题直接相关的句子、数据、定义、公式、结论
2. 删除无关的铺垫、背景、重复表述、页眉页脚、目录项等噪音
3. 保留原文的专业术语和关键表述，不 paraphrase、不改写原意
4. 可以删减句子，但不能合并或修改句子本身的意思
5. 按原文顺序输出，保留片段编号
6. 严格以JSON数组格式输出，每个元素对应一个片段的压缩结果
7. 如果某个片段完全无关，输出空字符串

输出格式示例：
["片段1压缩后的内容", "片段2压缩后的内容", "片段3压缩后的内容"]"""

        user_content = f"""用户问题：{query}

待压缩文档片段：
{docs_block}

请输出压缩结果JSON数组："""

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.1,
                stream=False
            )
            content = resp.choices[0].message.content.strip()

            # 解析JSON
            if content.startswith("```"):
                content = content.split("\n", 1)[-1] if "\n" in content else content
                content = content.rstrip("`").strip()

            compressed_list = json.loads(content)

            # 校验长度，异常则返回原文档
            if len(compressed_list) != len(documents):
                logger.warning("[ContextCompressor] 压缩结果长度不匹配，返回原文档")
                return documents

            # 重组结果，保留原文档所有元信息，只替换content
            result = []
            for i, doc in enumerate(documents):
                new_content = compressed_list[i].strip()
                if new_content:
                    new_doc = doc.copy()
                    new_doc["content"] = new_content
                    result.append(new_doc)

            logger.info(f"[ContextCompressor] 压缩完成: {len(documents)} -> {len(result)} 个有效片段")
            return result if result else documents

        except Exception as e:
            logger.warning(f"[ContextCompressor] 上下文压缩失败，返回原文档: {str(e)}")
            return documents

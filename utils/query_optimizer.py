"""
查询优化器模块
功能：
1. 查询改写（Query Rewriting）：将用户口语化/模糊问题改写成更适合检索的标准查询
2. 多查询扩展（Multi-Query Expansion）：生成多个语义相近的查询，并行检索扩大召回
设计原则：纯函数式，无状态，可独立测试，与检索器解耦
"""

from openai import OpenAI
from config.settings import settings
from config.logger import logger
from typing import List
import json


class QueryOptimizer:
    """查询优化器：改写 + 多查询扩展"""

    def __init__(self):
        self.client = OpenAI(
            api_key=settings.LLM_API_KEY,
            base_url=settings.LLM_BASE_URL
        )
        self.model = settings.LLM_MODEL_NAME

    # ==================== 1. 查询改写 ====================
    def rewrite_query(self, original_query: str, conversation_history: list = None) -> str:
        """
        将用户原始问题改写成更适合向量检索的标准查询
        - 去除口语化表达
        - 补全省略的上下文（结合对话历史）
        - 提取核心实体和关键词
        :param original_query: 用户原始问题
        :param conversation_history: 对话历史（可选，用于上下文补全）
        :return: 改写后的检索查询
        """
        history_text = ""
        if conversation_history:
            recent = conversation_history[-3:]
            history_lines = []
            for msg in recent:
                role = "用户" if msg.get("role") == "user" else "助手"
                history_lines.append(f"{role}：{msg.get('content', '')}")
            history_text = "\n".join(history_lines)

        system_prompt = """你是一个专业的检索查询改写助手。
任务：将用户的原始问题改写成更适合向量数据库检索的标准查询。
改写规则：
1. 去除口语化、情绪化表达，保留核心语义
2. 如果问题中有指代（"这个""它""上面说的"），结合对话历史补全具体指代对象
3. 提取问题中的核心实体、专业术语、关键词
4. 改写后的查询应该是一个完整、明确、语义清晰的问题或陈述
5. 只输出改写后的查询本身，不要任何解释、前缀或引号

示例：
用户："这东西咋用啊？"（历史中讨论过Chroma向量库）
改写：Chroma向量数据库的使用方法和基本操作

用户："杠杆平衡条件是什么"
改写：杠杆的平衡条件公式及原理"""

        user_content = f"对话历史：\n{history_text}\n\n用户原始问题：{original_query}\n\n请输出改写后的检索查询：" if history_text else f"用户原始问题：{original_query}\n\n请输出改写后的检索查询："

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.3,
                stream=False
            )
            rewritten = resp.choices[0].message.content.strip()
            # 清理可能的引号
            rewritten = rewritten.strip('"').strip("'").strip("「」").strip("『』")
            logger.info(f"[QueryOptimizer] 查询改写: '{original_query}' -> '{rewritten}'")
            return rewritten if rewritten else original_query
        except Exception as e:
            logger.warning(f"[QueryOptimizer] 查询改写失败，使用原始查询: {str(e)}")
            return original_query

    # ==================== 2. 多查询扩展 ====================
    def expand_multi_query(self, query: str, num_queries: int = 3) -> List[str]:
        """
        基于原始查询生成多个语义相近但角度不同的检索查询
        用于并行检索，扩大召回面，降低单一查询的漏检风险
        :param query: 原始查询（建议传入改写后的查询）
        :param num_queries: 生成的扩展查询数量（不含原始查询）
        :return: 包含原始查询在内的查询列表
        """
        system_prompt = f"""你是一个专业的检索查询扩展助手。
任务：基于用户给出的查询，生成 {num_queries} 个语义相近但表述角度不同的检索查询。
生成规则：
1. 每个查询都必须与原始查询语义高度相关，但表述方式、侧重点、关键词组合不同
2. 可以从不同角度切入：定义原理、操作方法、应用场景、公式计算、对比区别等
3. 每个查询都是独立完整的，适合单独用于向量检索
4. 不要生成与原始查询完全相同的查询
5. 严格以JSON数组格式输出，只输出JSON，不要任何解释文字

输出格式示例：
["查询1的内容", "查询2的内容", "查询3的内容"]"""

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"原始查询：{query}\n\n请生成扩展查询："}
                ],
                temperature=0.5,
                stream=False
            )
            content = resp.choices[0].message.content.strip()

            # 解析JSON数组（兼容可能的markdown代码块）
            if content.startswith("```"):
                content = content.split("\n", 1)[-1] if "\n" in content else content
                content = content.rstrip("`").strip()

            expanded = json.loads(content)
            if not isinstance(expanded, list):
                raise ValueError("返回结果不是列表")

            # 去重 + 过滤空值
            expanded = [q.strip() for q in expanded if q and q.strip()]
            expanded = list(dict.fromkeys(expanded))  # 保序去重

            # 合并原始查询，去重
            all_queries = [query] + [q for q in expanded if q != query]
            all_queries = all_queries[:num_queries + 1]  # 控制总数

            logger.info(f"[QueryOptimizer] 多查询扩展: 生成 {len(all_queries)} 个查询")
            return all_queries

        except Exception as e:
            logger.warning(f"[QueryOptimizer] 多查询扩展失败，使用单查询: {str(e)}")
            return [query]

    # ==================== 3. 组合入口 ====================
    def optimize(self, query: str, conversation_history: list = None,
                 enable_rewrite: bool = True, enable_multi_query: bool = False,
                 num_queries: int = 3) -> List[str]:
        """
        查询优化组合入口
        :param query: 用户原始问题
        :param conversation_history: 对话历史
        :param enable_rewrite: 是否启用查询改写
        :param enable_multi_query: 是否启用多查询扩展
        :param num_queries: 多查询扩展数量
        :return: 优化后的查询列表（至少包含1个查询）
        """
        # Step 1: 查询改写
        if enable_rewrite:
            base_query = self.rewrite_query(query, conversation_history)
        else:
            base_query = query

        # Step 2: 多查询扩展
        if enable_multi_query:
            return self.expand_multi_query(base_query, num_queries)
        else:
            return [base_query]

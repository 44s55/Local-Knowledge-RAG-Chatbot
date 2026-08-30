def build_rag_prompt(query: str, context: str, history: list = None) -> list:
    """构建RAG问答的完整prompt"""
    system_prompt = """你是一个专业的知识库问答助手。请严格遵循以下规则：

1. 只根据下面提供的【文档内容】回答问题
2. 如果文档内容不足以回答问题，请明确说"根据已有文档，无法回答此问题"，不要编造
3. 回答时尽量引用文档中的原文，保持准确性
4. 回答要简洁清晰，使用中文"""

    messages = [{"role": "system", "content": system_prompt}]

    if history:
        messages.extend(history)

    user_message = f"""【文档内容】
{context}

【用户问题】
{query}

请基于以上文档内容回答问题。"""

    messages.append({"role": "user", "content": user_message})
    return messages


def build_context_from_results(results: list) -> str:
    """将检索结果拼接成上下文字符串"""
    parts = []
    for i, r in enumerate(results, 1):
        parts.append(f"【片段{i}】来源：{r['source']}\\\\n{r['content']}")
    return "\\\\n\\\\n".join(parts)
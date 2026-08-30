# run_llm.py
from openai import OpenAI
from utils.config import LLM_API_KEY, LLM_API_BASE, LLM_MODEL

def chat_with_llm(messages: list, temperature: float = 0.3) -> str:
    """调用大模型对话"""
    client = OpenAI(
        api_key=LLM_API_KEY,
        base_url=LLM_API_BASE,
    )
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=temperature,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[LLM调用失败] {str(e)}"

if __name__ == "__main__":
    test_messages = [{"role": "user", "content": "用一句话介绍RAG技术"}]
    result = chat_with_llm(test_messages)
    print("模型回复：", result)

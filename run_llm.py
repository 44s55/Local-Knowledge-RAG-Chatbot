# run_llm.py
from openai import OpenAI
from utils.config import settings

def chat_with_llm(messages: list, temperature: float = 0.3) -> str:
    """调用大模型对话，复用项目原有Settings配置"""
    client = OpenAI(
        api_key=settings.LLM_API_KEY,
        base_url=settings.LLM_BASE_URL,
    )
    try:
        response = client.chat.completions.create(
            model=settings.LLM_MODEL_NAME,
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

from utils.rag_chain import rag_chain_instance

embedder = rag_chain_instance.embedder

text = "RAG测试文本，用于验证嵌入模型"

vec = embedder.embed_text(text)
print(f"单条向量结果：{vec}")
print(f"向量长度：{len(vec) if vec else 0}")

vecs = embedder.embed_texts([text, "第二条测试文字"])
print(f"批量向量：{vecs}")
print(f"批量数量：{len(vecs)}")

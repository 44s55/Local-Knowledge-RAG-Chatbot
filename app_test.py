import sys
sys.path.append(".")
print("测试脚本启动成功")

from utils.retriever import Retriever
print("导入Retriever成功")

ret = Retriever()
ctx, src = ret.get_context("什么是RAG", top_k=1)
print("检索上下文：")
print(ctx)

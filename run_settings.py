# test_settings.py
from utils.config import settings

print(f"重排开关 RERANK_ENABLE = {settings.RERANK_ENABLE}")
print(f"是否使用云端重排 RERANK_USE_CLOUD = {settings.RERANK_USE_CLOUD}")
print(f"重排后保留片段数 RERANK_TOP_K = {settings.RERANK_TOP_K}")
print(f"重排过滤阈值 RERANK_THRESHOLD = {settings.RERANK_THRESHOLD}")

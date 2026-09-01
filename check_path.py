# check_path.py
import sys
print("=====完整sys.path列表====")
for idx,p in enumerate(sys.path):
    print(f"{idx:2d}: {p}")

from pathlib import Path
proj = Path(r"D:\AI_Projects\Local-Knowledge-RAG-Chatbot")
utils_path = proj / "utils"
print(f"\nutils文件夹完整路径: {utils_path}")
print(f"utils是否存在: {utils_path.exists()}")
print(f"utils下文件列表: {list(utils_path.glob('*.py'))}")

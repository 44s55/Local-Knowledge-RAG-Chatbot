# check_key.py
from dotenv import load_dotenv
res = load_dotenv()
print(f"load_dotenv返回:{res}")

import os
key = os.getenv("DASHSCOPE_API_KEY")
if key:
    print(f"拿到密钥，前缀：{key[:8]}***")
else:
    print("密钥=None，读取失败")

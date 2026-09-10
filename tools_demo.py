# 直接从文件导入，不经过 __init__.py
import sys
sys.path.append(".")

from tool.calculator_tool import calculate_expression
from tool.datetime_tool import datetime_tool

print("=== 测试计算器工具 ===")
res = calculate_expression("(10 + 20) * 3")
print(res)

print("\n=== 测试时间工具 ===")
res = datetime_tool()
print(res)

print("\n✅ 工具模块加载正常")

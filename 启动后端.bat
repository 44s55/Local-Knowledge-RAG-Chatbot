@echo off
chcp 65001 > nul
echo ========================================
echo  RAG知识库问答系统 - 后端服务
echo ========================================
echo 正在启动后端服务...
echo 服务地址: http://127.0.0.1:8000
echo 接口文档: http://127.0.0.1:8000/docs
echo ========================================
D:\conda\envs\llm_env\python.exe main_fastapi.py
pause

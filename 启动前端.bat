@echo off
chcp 65001 > nul
echo ========================================
echo  RAG知识库问答系统 - 前端页面
echo ========================================
echo 正在启动前端页面...
echo 访问地址: http://127.0.0.1:7860
echo ========================================
D:\conda\envs\llm_env\python.exe web_gradio.py
pause

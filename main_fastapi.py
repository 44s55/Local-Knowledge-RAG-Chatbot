# 屏蔽Chroma遥测冗余报错（环境变量方式，全版本兼容）
import os
os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"

from fastapi.responses import StreamingResponse
import json
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict
import tempfile
from pathlib import Path

from config.settings import settings
from config.logger import logger
from utils.exceptions import FileNotSupportError, KnowledgeBaseEmptyError, LLMRequestError
from ingest.ingest_pipeline import IngestPipeline
from utils.simple_agent import SimpleLightAgent
from utils.rag_chain import RAGChain

app = FastAPI(title="本地知识库RAG问答系统 - 后端接口")
logger.info("✅ FastAPI RAG后端服务准备启动")

# 全局初始化RAG链路 + Agent，服务启动时加载一次
rag_chain = RAGChain()
agent = SimpleLightAgent(rag_chain=rag_chain)


# ===================== 请求体定义 =====================
class ChatRequest(BaseModel):
    question: str
    history: List[Dict] = []
    top_k: int = 4
    enable_rerank: bool = False
    # ===== 新增：检索优化开关（和前端payload字段一一对应）=====
    enable_query_rewrite: bool = False   # 查询改写
    enable_multi_query: bool = False     # 多查询扩展
    enable_compression: bool = False     # 上下文压缩


# ===================== 全局异常处理器（兜底） =====================
@app.exception_handler(FileNotSupportError)
async def file_not_support_handler(request: Request, exc: FileNotSupportError):
    logger.warning(f"文件格式不支持: {str(exc)}")
    return JSONResponse(
        status_code=200,
        content={"code": 400, "msg": str(exc), "data": None}
    )


@app.exception_handler(KnowledgeBaseEmptyError)
async def knowledge_empty_handler(request: Request, exc: KnowledgeBaseEmptyError):
    logger.warning(f"知识库为空: {str(exc)}")
    return JSONResponse(
        status_code=200,
        content={"code": 400, "msg": str(exc), "data": None}
    )


@app.exception_handler(LLMRequestError)
async def llm_request_handler(request: Request, exc: LLMRequestError):
    logger.error(f"大模型调用失败: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=200,
        content={"code": 500, "msg": "大模型服务调用失败，请稍后重试", "data": None}
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"接口全局异常: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=200,
        content={"code": 500, "msg": "系统内部异常，请稍后重试", "data": None}
    )


# ===================== 接口定义 =====================
@app.post("/upload", summary="上传文档并解析入库")
async def upload_document(file: UploadFile = File(...)):
    logger.info(f"收到上传文件请求：{file.filename}")
    try:
        # 1. 校验文件格式
        allow_suffix = {".pdf", ".md", ".txt"}
        suffix = Path(file.filename).suffix.lower()
        if suffix not in allow_suffix:
            raise FileNotSupportError(f"不支持的文件格式：{suffix}，仅支持PDF/MD/TXT")

        # 2. 保存临时文件（Windows兼容：关闭句柄后再读取）
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
            tmp.close()

        # 3. 调用入库流水线
        pipeline = IngestPipeline()
        pipeline.process_file(tmp_path)
        pipeline.retriever.rebuild_bm25()  # 重建BM25稀疏索引

        # 删除临时文件
        os.unlink(tmp_path)

        logger.info(f"文件 {file.filename} 解析入库成功")
        return {
            "code": 0,
            "msg": "文档上传解析成功",
            "data": {"filename": file.filename}
        }

    except FileNotSupportError as e:
        logger.warning(str(e))
        return {"code": 400, "msg": str(e)}
    except Exception as e:
        logger.error(f"上传解析异常：{str(e)}", exc_info=True)
        return {"code": 500, "msg": "文档解析失败，请检查文件格式或重试"}


@app.post("/chat", summary="知识库问答接口")
async def chat(req: ChatRequest):
    logger.info(f"用户提问：{req.question}")
    try:
        # 调用Agent完整链路
        answer, reference_docs, decision = agent.run(
            user_query=req.question,
            top_k=req.top_k,
            enable_rerank=req.enable_rerank,
            conversation_history=req.history,
            # ===== 新增：传递三个优化开关 =====
            enable_query_rewrite=req.enable_query_rewrite,
            enable_multi_query=req.enable_multi_query,
            enable_compression=req.enable_compression
        )

        return {
            "code": 0,
            "msg": "success",
            "data": {
                "answer": answer,
                "reference": reference_docs,
                "decision": decision
            }
        }

    except KnowledgeBaseEmptyError as e:
        logger.warning(str(e))
        return {"code": 400, "msg": "当前知识库为空，请先上传文档"}
    except LLMRequestError as e:
        logger.error(f"大模型调用失败：{str(e)}")
        return {"code": 500, "msg": "大模型服务调用失败，请稍后重试"}
    except Exception as e:
        logger.error(f"问答异常：{str(e)}", exc_info=True)
        return {"code": 500, "msg": "系统内部异常，请稍后重试"}


@app.post("/chat/stream", summary="流式知识库问答接口（SSE）")
async def chat_stream(req: ChatRequest):
    logger.info(f"收到流式提问：{req.question}")
    try:
        # 1. 复用原有Agent完整逻辑：意图判断 + 文档检索
        # 这里会得到和普通接口完全一致的 decision、reference_docs，保证溯源格式统一
        _, reference_docs, decision = agent.run(
            user_query=req.question,
            top_k=req.top_k,
            enable_rerank=req.enable_rerank,
            conversation_history=req.history,
            # ===== 新增：传递三个优化开关 =====
            enable_query_rewrite=req.enable_query_rewrite,
            enable_multi_query=req.enable_multi_query,
            enable_compression=req.enable_compression
        )

        # 2. 流式生成器
        def generate():
            # 第一帧：返回元数据（检索结果、决策），和普通接口格式完全一致
            yield f"data: {json.dumps({'type': 'meta', 'decision': decision, 'reference': reference_docs}, ensure_ascii=False)}\n\n"

            # 第二帧开始：逐字返回回答内容
            if decision.get("tool") == "knowledge_search" and reference_docs:
                for text_chunk in rag_chain.stream_chat(
                        query=req.question,
                        context_docs=reference_docs,
                        conversation_history=req.history
                ):
                    yield f"data: {json.dumps({'type': 'text', 'content': text_chunk}, ensure_ascii=False)}\n\n"

            elif decision.get("tool") == "no_tool":
                # 闲聊场景也走流式
                for text_chunk in rag_chain.stream_chat(
                        query=req.question,
                        context_docs=[],
                        conversation_history=req.history
                ):
                    yield f"data: {json.dumps({'type': 'text', 'content': text_chunk}, ensure_ascii=False)}\n\n"

            else:
                yield f"data: {json.dumps({'type': 'text', 'content': '知识库中未查询到相关内容'}, ensure_ascii=False)}\n\n"

            # 结束标记
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
        )

    except Exception as e:
        logger.error(f"流式问答异常：{str(e)}", exc_info=True)
        return JSONResponse(
            status_code=200,
            content={"code": 500, "msg": "流式生成失败", "data": None}
        )


@app.post("/clear_history", summary="清空对话上下文")
async def clear_history():
    logger.info("请求清空对话历史")
    return {"code": 0, "msg": "对话历史已清空"}


# ===================== 启动入口 =====================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=settings.FASTAPI_HOST,
        port=settings.FASTAPI_PORT
    )

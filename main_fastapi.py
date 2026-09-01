from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List, Dict
import os
import tempfile
from pathlib import Path

from config.settings import settings
from utils.logger import logger
from utils.exceptions import FileNotSupportError, KnowledgeBaseEmptyError, LLMRequestError

from ingest.ingest_pipeline import IngestPipeline
from utils.simple_agent import SimpleLightAgent
from utils.rag_chain import RAGChain

app = FastAPI(title="本地知识库RAG问答系统 - 后端接口")

# 全局初始化RAG链路 + Agent，服务启动时加载一次
rag_chain = RAGChain()
agent = SimpleLightAgent(rag_chain=rag_chain)


# ===================== 请求体定义 =====================
class ChatRequest(BaseModel):
    question: str
    history: List[Dict] = []
    top_k: int = 4
    enable_rerank: bool = False


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
            conversation_history=req.history
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

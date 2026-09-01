# utils/config.py
import os
from dotenv import load_dotenv
VECTOR_DB_DIR: str

'''
库：python‑dotenv
需要pip安装：pip install python‑dotenv
作用：读取项目根目录下的 .env 文件，把密钥、配置加载进环境变量
企业规范：密钥绝不硬编码写死在py代码里面；切换开发/生产环境，只修改.env，不动源码
load_dotenv() 默认会去项目根目录找 .env
'''

# 加载根目录的.env配置文件
load_dotenv()


class Settings:
    """全局配置类，整个项目统一读取参数，一处修改全局生效"""

    # ========== 大模型 LLM 配置（阿里云百炼兼容OpenAI接口） ==========
    LLM_API_KEY: str = os.getenv("LLM_API_KEY")
    LLM_MODEL_NAME: str = os.getenv("LLM_MODEL_NAME")
    LLM_BASE_URL: str = os.getenv("LLM_BASE_URL")

    # ========== 嵌入模型 & 重排序模型配置 ==========
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME")
    RERANK_MODEL_NAME: str = os.getenv("RERANK_MODEL_NAME")

    # ========== 向量数据库参数 ==========
    VECTOR_DB_PATH: str = os.getenv("VECTOR_DB_PATH")
    TOP_K_VECTOR: int = int(os.getenv("TOP_K_VECTOR"))
    TOP_K_BM25: int = int(os.getenv("TOP_K_BM25"))
    RERANK_TOP_N: int = int(os.getenv("RERANK_TOP_N"))
    # RAG幻觉防护：向量距离阈值，大于该值代表知识库相关性很低，拒绝回答
    RAG_DISTANCE_THRESHOLD: float = 1.2

    # ========== 文本切分Chunk参数 ==========
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP"))


# 全局单例对象，别的文件导入： from utils.config import settings
settings = Settings()

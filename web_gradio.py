"""
RAG知识库网页交互界面，基于Gradio4.x实现
功能：
1. 上传PDF/TXT/MD文档，保存到本地data目录，自动执行ingest流水线完成切片向量化持久入库
2. 多轮对话问答，复用ConversationMemory记忆模块
3. 展示溯源信息：来源文件、向量距离、重排分数、检索片段
4. 支持参数调节：检索top_k
5. 支持清空对话会话
6. 新增：清空全部知识库按钮
7. 新增：Reranker重排开关，每次提问实时生效
8. 新增：Agent决策过程可视化展示
"""
import os
import query

# 全局关闭所有遥测开关
os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"
os.environ["ANONYMIZED_TELEMETRY"] = "false"
os.environ["GRADIO_ANALYTICS_ENABLED"] = "0"
os.environ["GRADIO_API_DOCS"] = "0"

# 提前导入chroma并强制拦截遥测捕获函数，彻底消除参数报错
import chromadb

try:
    # 直接替换遥测核心捕获函数，无视参数数量，直接返回空
    import chromadb.telemetry.events

    chromadb.telemetry.events.capture = lambda *args, **kwargs: None
except Exception:
    try:
        # 兼容其他版本的模块路径
        import chromadb.telemetry

        chromadb.telemetry.capture = lambda *args, **kwargs: None
    except Exception:
        pass

from pathlib import Path
from ingest.ingest_pipeline import IngestPipeline

# 猴子补丁，捕获异常防止模块缺失崩溃
try:
    import gradio_client.utils as gcu

    original_json_schema_to_python_type = gcu._json_schema_to_python_type


    def patched_json_schema_to_python_type(schema, defs):
        if not isinstance(schema, dict):
            return "any"
        return original_json_schema_to_python_type(schema, defs)


    gcu._json_schema_to_python_type = patched_json_schema_to_python_type

    original_get_type = gcu.get_type


    def patched_get_type(schema):
        if not isinstance(schema, dict):
            return None
        return original_get_type(schema)


    gcu.get_type = patched_get_type
except Exception:
    pass

# ========= 业务导入 =========
import gradio as gr
import shutil
from dotenv import load_dotenv
from utils.rag_chain import RAGChain
from utils.config import settings
from utils.conversation_memory import ConversationMemory
from utils.simple_agent import SimpleLightAgent

# 加载.env环境变量
load_dotenv()

# ---------------------- 全局单例对象 ----------------------
# 和ingest_pipeline完全统一：自动推导项目根目录，杜绝相对路径bug
SCRIPT_FILE = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_FILE.parent
DATA_FOLDER = PROJECT_ROOT / "data"

# 初始化对话记忆，最大保存6轮对话
memory = ConversationMemory(max_turns=6)
# RAG主业务链路，内部已经封装retriever + reranker
rag_chain = RAGChain()
# 轻量化Agent调度层，复用已有RAG实例
agent = SimpleLightAgent(rag_chain=rag_chain)
# 初始化ingest流水线实例（修复重复实例化问题）
ingest_pipeline = IngestPipeline()


def upload_files_to_data(files):
    """
    gradio上传文件回调函数
    将网页上传的文件拷贝到项目data目录，自动调用ingest流水线完成切片、向量化、持久入库
    :param files: gradio上传得到的文件列表
    :return: str，返回处理结果提示文本
    """
    if not files:
        return "⚠️未检测到上传文件，请选择PDF/TXT/MD文档"

    os.makedirs(DATA_FOLDER, exist_ok=True)
    success_count = 0
    for file in files:
        filename = os.path.basename(file.name)
        suffix = Path(filename).suffix.lower()
        # 过滤不支持后缀
        if suffix not in (".txt", ".pdf", ".md"):
            continue
        target_path = str(DATA_FOLDER / filename)
        shutil.copy(file, target_path)
        # 单文件执行流水线处理：切片+向量化+入库
        ingest_pipeline.process_file(target_path)
        success_count += 1
    # 全部上传完成后重建BM25索引
    ingest_pipeline.retriever.rebuild_bm25()
    return f"✅成功处理 {success_count} 个文档，已完成切片&向量入库，BM25索引已更新。"


def convert_openai_history_to_gradio(openai_msg_list):
    """把 [{"role":"user","content":"xx"},...] 转成 gradio chatbot [[u,b],[u,b]]"""
    gradio_history = []
    temp_user = None
    for msg in openai_msg_list:
        r = msg["role"]
        c = msg["content"]
        if r == "user":
            temp_user = c
        elif r == "assistant":
            gradio_history.append([temp_user, c])
            temp_user = None
    return gradio_history

def chat_handle_message(user_message, chat_history, top_k_slider, rerank_switch):
    # 形参统一转内部变量
    user_query = user_message
    top_k = int(top_k_slider)
    enable_rerank = rerank_switch

    # ========== 把Gradio历史转为OpenAI格式 ==========
    conversation_history = []
    for user_msg, bot_msg in chat_history:
        if user_msg:
            conversation_history.append({"role": "user", "content": user_msg})
        if bot_msg:
            conversation_history.append({"role": "assistant", "content": bot_msg})
    # 只保留最近6轮，控制token
    if len(conversation_history) > 12:
        conversation_history = conversation_history[-12:]

    # 调用 Agent 执行完整问答，传入历史
    answer, sources, decision = agent.run(
        user_query=user_query,
        top_k=top_k,
        enable_rerank=enable_rerank,
        conversation_history=conversation_history
    )

    # 追加到对话历史
    chat_history.append((user_message, answer))

    # ========== 格式化 Agent 决策展示 ==========
    tool = decision.get("tool", "未知")
    thought = decision.get("thought", "无")
    agent_text = f"""**思考原因**：{thought}

**选择工具**：{tool}"""

    # ========== 格式化溯源信息输出 ==========
    source_text = ""
    if not sources:
        source_text = "本次未调用知识库检索，无溯源信息。"
    else:
        for idx, doc in enumerate(sources):
            metadata = doc.get("metadata", {})
            source_text += f"【片段{idx + 1}】\n"
            source_text += f"文件：{metadata.get('source', '未知')}\n"
            distance = metadata.get("distance", 'N/A')
            source_text += f"距离：{distance:.4f}\n" if isinstance(distance, float) else f"距离：{distance}\n"
            rerank_score = metadata.get('rerank_score')
            source_text += f"重排分数：{rerank_score if rerank_score is not None else '未开启'}\n"
            source_text += f"内容：{doc.get('content', '')[:120]}...\n\n"

    # 统一返回3个结果
    return chat_history, agent_text, source_text


def clear_chat_session():
    """清空对话会话，同时清空Agent决策展示和溯源展示"""
    memory.clear()
    # 返回三个值，分别对应聊天窗口、Agent决策框、溯源框
    return [], "等待提问...", "对话已清空，请发起新问题"


def handle_clear_kb():
    """
    清空全部知识库：清空向量库集合、清空BM25索引，保留聊天会话记录
    :return: 字符串提示，输出到溯源详情文本框
    """
    rag_chain.vector_store.clear()
    rag_chain.hybrid_retriever.clear_bm25()
    # 返回提示，只更新溯源面板，聊天窗口内容不动
    return "✅向量库与BM25索引已全部清空，请重新上传文档！"


# ---------------------- Gradio UI页面布局定义 ----------------------
with gr.Blocks(title="本地RAG知识库问答系统") as demo:
    gr.Markdown("# 📚 Local‑Knowledge‑RAG‑Chatbot 网页问答端")

    with gr.Row():
        # 左侧面板：上传、参数、操作按钮
        with gr.Column(scale=1):
            gr.Markdown("## 📂文档上传")
            upload_file_input = gr.File(label="上传知识库文档", file_types=[".pdf", ".txt", ".md"],
                                        file_count="multiple")
            upload_run_btn = gr.Button("💾保存文件到data目录并入库")
            upload_status_text = gr.Textbox(label="上传状态", interactive=False)

            gr.Markdown("## ⚙️检索参数")
            top_k_slider = gr.Slider(minimum=2, maximum=10, value=4, step=1, label="Top‑K 检索片段数")
            # Reranker重排开关
            rerank_checkbox = gr.Checkbox(
                label="开启Reranker重排",
                value=False,
                info="开启后对召回片段做精排，提升答案匹配度"
            )

            gr.Markdown("## 🧹操作")
            clear_chat_btn = gr.Button("清空对话会话")
            btn_clear_kb = gr.Button("⚠️清空全部知识库", variant="stop")

        # 右侧面板：聊天窗口 + Agent决策 + 溯源详情
        with gr.Column(scale=2):
            chatbot_ui = gr.Chatbot(label="问答对话窗口", height=520)
            user_input_box = gr.Textbox(label="请输入你的问题", placeholder="基于知识库提问...")

            # 新增：Agent决策过程展示框
            agent_thought_box = gr.Textbox(label="🤖 Agent决策过程", value="等待提问...", interactive=False, lines=3)

            source_detail_box = gr.Textbox(label="📚 检索溯源详情", interactive=False, lines=14)

    # ========== 绑定事件回调 ==========
    upload_run_btn.click(
        fn=upload_files_to_data,
        inputs=[upload_file_input],
        outputs=[upload_status_text]
    )

    # 回车提交提问：输出对应3个组件
    user_input_box.submit(
        fn=chat_handle_message,
        inputs=[user_input_box, chatbot_ui, top_k_slider, rerank_checkbox],
        outputs=[chatbot_ui, agent_thought_box, source_detail_box]
    )

    # 清空对话：同步清空3个输出组件
    clear_chat_btn.click(
        fn=clear_chat_session,
        inputs=[],
        outputs=[chatbot_ui, agent_thought_box, source_detail_box]
    )

    # 清空知识库：只更新溯源提示
    btn_clear_kb.click(
        fn=handle_clear_kb,
        inputs=[],
        outputs=[source_detail_box]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_api=False
    )

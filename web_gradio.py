"""
RAG知识库网页交互界面，基于Gradio4.x实现
功能：
1. 上传PDF/TXT/DOCX文档，保存到本地data目录（注：暂未自动执行向量化入库，需要后续实现data文档加载逻辑）
2. 多轮对话问答，复用ConversationMemory记忆模块
3. 展示溯源信息：来源文件、检索片段
4. 支持参数调节：检索top_k
5. 支持清空对话会话
"""
import os
# 关闭gradio遥测，减少无关报错
os.environ["GRADIO_ANALYTICS_ENABLED"] = "0"
os.environ["GRADIO_API_DOCS"] = "0"

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

# 导入项目内部已封装模块
from utils.rag_chain import RAGChain
from utils.config import settings
from utils.conversation_memory import ConversationMemory

# 加载.env环境变量
load_dotenv()

# ---------------------- 全局单例对象 ----------------------
# data知识库目录
DATA_FOLDER = "./data"
# 初始化对话记忆，最大保存6轮对话
memory = ConversationMemory(max_turns=6)
# RAG主业务链路，内部已经封装retriever
rag_chain = RAGChain()


def upload_files_to_data(files):
    """
    gradio上传文件回调函数
    将网页上传的文件拷贝到项目data目录
    注意：当前ingest.py仅内置测试文本，**不会自动对data目录文件做切分向量化入库**，入库逻辑待实现
    :param files: gradio上传得到的文件列表
    :return: str，返回处理结果提示文本
    """
    if not files:
        return "⚠️未检测到上传文件，请选择PDF/TXT/DOCX文档"

    os.makedirs(DATA_FOLDER, exist_ok=True)

    for file in files:
        filename = os.path.basename(file.orig_name)
        target_path = os.path.join(DATA_FOLDER, filename)
        shutil.copy(file, target_path)

    return f"✅成功保存 {len(files)} 个文件到./data目录。提示：尚未执行文档向量化入库，请补充data目录加载逻辑。"


def chat_handle_message(user_query: str, history, top_k: int):
    """
    聊天问答回调函数，对接RAG完整业务链路
    :param user_query: 用户输入的问题
    :param history: gradio聊天历史
    :param top_k: 检索返回候选片段数量
    :return: 更新后的聊天历史，溯源详情文本
    """
    if not user_query or user_query.strip() == "":
        return history, "⚠️问题不能为空"

    memory.add_user_message(user_query)
    answer, source_docs = rag_chain.invoke(user_query, top_k=top_k)
    memory.add_assistant_message(answer)

    source_info = "=====📑检索溯源信息=====\n"
    if len(source_docs) == 0:
        source_info += "本次没有检索到相关知识库片段\n"
    else:
        for idx, doc in enumerate(source_docs):
            source_info += f"\n【片段{idx+1}】\n"
            if isinstance(doc, dict):
                meta = doc.get("metadata", {})
                page_content = doc.get("content", "")
            else:
                meta = getattr(doc, "metadata", {})
                page_content = getattr(doc, "page_content", "")

            source_info += f"来源文件：{meta.get('source', '未知')}\n"
            source_info += f"重排分数：{meta.get('rerank_score', '--')}\n"
            source_info += f"片段内容：{page_content[:350]}...\n"

    history.append([user_query, answer])
    return history, source_info


def clear_chat_session():
    """
    清空对话会话：清空内存对话记忆，清空前端聊天记录
    :return: 空聊天列表，重置溯源面板提示
    """
    memory.clear()
    return [], "对话已清空，请发起新问题"


# ---------------------- Gradio UI页面布局定义 ----------------------
with gr.Blocks(title="本地RAG知识库问答系统") as demo:
    gr.Markdown("# 📚 Local‑Knowledge‑RAG‑Chatbot 网页问答端")

    with gr.Row():
        # 左侧面板：上传、参数、操作按钮
        with gr.Column(scale=1):
            gr.Markdown("## 📂文档上传")
            upload_file_input = gr.File(label="上传知识库文档", file_types=[".pdf", ".txt", ".docx"], file_count="multiple")
            upload_run_btn = gr.Button("💾保存文件到data目录")
            upload_status_text = gr.Textbox(label="上传状态", interactive=False)

            gr.Markdown("## ⚙️检索参数")
            top_k_slider = gr.Slider(minimum=2, maximum=10, value=4, step=1, label="Top‑K 检索片段数")

            gr.Markdown("## 🧹操作")
            clear_chat_btn = gr.Button("清空对话会话")

        # 右侧面板：聊天窗口 + 溯源详情
        with gr.Column(scale=2):
            chatbot_ui = gr.Chatbot(label="问答对话窗口", height=520)
            user_input_box = gr.Textbox(label="请输入你的问题", placeholder="基于知识库提问...")
            source_detail_box = gr.Textbox(label="检索溯源详情", interactive=False, lines=14)

    # 绑定事件回调
    upload_run_btn.click(
        fn=upload_files_to_data,
        inputs=[upload_file_input],
        outputs=[upload_status_text]
    )

    user_input_box.submit(
        fn=chat_handle_message,
        inputs=[user_input_box, chatbot_ui, top_k_slider],
        outputs=[chatbot_ui, source_detail_box]
    )

    clear_chat_btn.click(
        fn=clear_chat_session,
        inputs=[],
        outputs=[chatbot_ui, source_detail_box]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_api=False
    )

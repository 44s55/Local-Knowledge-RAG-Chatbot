import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Optional


class ChatMemoryManager:
    def __init__(self, db_path: str = "chat_memory.db", max_rounds: int = 10, summary_threshold: int = 8):
        """
        :param db_path: 对话历史数据库文件
        :param max_rounds: 最多保留多少轮原始对话
        :param summary_threshold: 超过多少轮之后触发历史摘要压缩
        """
        self.db_path = db_path
        self.max_rounds = max_rounds
        self.summary_threshold = summary_threshold
        self._init_db()

    def _init_db(self):
        """初始化数据库表"""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cursor = conn.cursor()
        # 会话信息
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            tenant_id TEXT DEFAULT "default",
            title TEXT,
            created_at TIMESTAMP
        )
        ''')
        # 单条对话记录
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            msg_id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            role TEXT,
            content TEXT,
            created_at TIMESTAMP,
            FOREIGN KEY(session_id) REFERENCES sessions(session_id)
        )
        ''')
        # 长对话摘要存储
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS session_summary (
            session_id TEXT PRIMARY KEY,
            summary_text TEXT,
            update_time TIMESTAMP,
            FOREIGN KEY(session_id) REFERENCES sessions(session_id)
        )
        ''')
        conn.commit()
        conn.close()

    def create_session(self, session_id: str, tenant_id: str = "default", title: str = ""):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cur = conn.cursor()
        cur.execute("INSERT OR IGNORE INTO sessions(session_id, tenant_id, title, created_at) VALUES (?,?,?,?)",
                    (session_id, tenant_id, title, datetime.now()))
        conn.commit()
        conn.close()

    def add_message(self, session_id: str, role: str, content: str):
        """新增单条对话，role取值 user / assistant"""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cur = conn.cursor()
        cur.execute("INSERT INTO messages(session_id, role, content, created_at) VALUES (?,?,?,?)",
                    (session_id, role, content, datetime.now()))
        conn.commit()
        conn.close()

    def _get_raw_messages(self, session_id: str) -> List[Dict]:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cur = conn.cursor()
        res = cur.execute("SELECT role,content FROM messages WHERE session_id=? ORDER BY created_at ASC", (session_id,))
        rows = res.fetchall()
        conn.close()
        return [{"role": r[0], "content": r[1]} for r in rows]

    def _generate_summary(self, msg_list: List[Dict], llm) -> str:
        prompt = "将下面多轮对话进行精简摘要，保留关键问题、上下文约束，字数控制在250字以内：\n{}"
        raw_text = "\n".join([f"{m['role']}:{m['content']}" for m in msg_list])
        resp = llm.invoke(prompt.format(raw_text))
        return resp.content

    def get_formatted_context(self, session_id: str, llm) -> str:
        """对外接口：获取处理完毕的上下文，超长自动摘要，仅返回需要送入大模型的文本"""
        msg_list = self._get_raw_messages(session_id)
        if len(msg_list) <= self.summary_threshold:
            # 轮数较少，直接原样拼接
            return "\n".join([f"{m['role']}：{m['content']}" for m in msg_list[-self.max_rounds:]])
        else:
            # 超过阈值，前面内容做摘要，保留最近若干轮完整对话
            old_msgs = msg_list[:-4]
            recent_msgs = msg_list[-4:]
            summary = self._generate_summary(old_msgs, llm)
            recent_text = "\n".join([f"{m['role']}：{m['content']}" for m in recent_msgs])
            final_ctx = f"【对话摘要】：{summary}\n【近期对话】\n{recent_text}"
            return final_ctx


# 单独测试入口
if __name__ == "__main__":
    print("模块加载正常，可以接入主程序。")
    print("使用示例：")
    print("manager = ChatMemoryManager()")
    print("manager.create_session(session_id=\"test001\")")

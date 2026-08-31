from typing import List, Dict


class ConversationMemory:
    """
    对话上下文记忆模块
    存储多轮对话历史，支持消息追加、获取历史、清空、截断历史，避免上下文溢出
    """
    def __init__(self, max_turns: int = 6):
        """
        :param max_turns: 最大保存多少轮对话(一轮=用户提问+模型回答)
        """
        self.max_turns = max_turns
        self.history: List[Dict[str, str]] = []

    def add_user_message(self, content: str) -> None:
        """新增用户消息"""
        self.history.append({"role": "user", "content": content})
        self._truncate()

    def add_assistant_message(self, content: str) -> None:
        """新增模型回复消息"""
        self.history.append({"role": "assistant", "content": content})
        self._truncate()

    def get_history(self) -> List[Dict[str, str]]:
        """获取完整对话历史，直接传给大模型messages参数"""
        return self.history.copy()

    def clear(self) -> None:
        """清空全部对话历史，开启新会话"""
        self.history.clear()

    def _truncate(self) -> None:
        """
        内部截断逻辑：限制最大轮数
        如果超过max_turns轮，丢弃最前面旧对话，保留后面最新对话
        每一轮包含 user + assistant 两条消息，总条数 = 2 * max_turns
        """
        max_length = self.max_turns * 2
        if len(self.history) > max_length:
            # 删掉头部多余消息，保留后面最新消息
            drop_count = len(self.history) - max_length
            self.history = self.history[drop_count:]


if __name__ == "__main__":
    # 本地自测，不需要网络，直接运行
    memory = ConversationMemory(max_turns=2)
    memory.add_user_message("什么是RAG？")
    memory.add_assistant_message("RAG是检索增强生成技术。")

    memory.add_user_message("它由哪些模块组成？")
    memory.add_assistant_message("加载、切片、向量化、检索、重排、生成。")

    memory.add_user_message("重排的作用是什么")
    memory.add_assistant_message("对检索出来的文档做二次相关性打分。")

    print("===当前对话历史===")
    for msg in memory.get_history():
        print(f"{msg['role']}: {msg['content']}")

    print("\n===清空会话===")
    memory.clear()
    print(memory.get_history())


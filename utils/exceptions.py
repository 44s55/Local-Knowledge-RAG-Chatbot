class FileNotSupportError(Exception):
    """不支持的文件格式异常"""
    pass


class KnowledgeBaseEmptyError(Exception):
    """知识库为空异常"""
    pass


class LLMRequestError(Exception):
    """大模型调用异常"""
    pass

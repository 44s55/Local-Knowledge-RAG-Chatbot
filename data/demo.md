Gitee - 基于 Git 的代码托管和研发协作平台 
我的
 
    
【大赛通知】开源中国「2026上海开源软件应用创新大赛」火热报名中，百万奖池等你的项目
richgiteeai
hx44s55
/
Local-Knowledge-RAG-Chatbot
你当前开源项目尚未选择许可证（LICENSE），点此选择并创建开源许可
 
 
44s55- 44s55 fix:修复ingest路径问题、向量入库空embedding异常，增加空文档防护 ed0fbbe 8分钟前
 12 次提交
data
 
fix:屏蔽chromadb遥测版本冲突日志，移除调试打印，系统稳定可用
 2天前
ingest
 
fix:修复ingest路径问题、向量入库空embedding异常，增加空文档防护
 8分钟前
loaders
 
fix:屏蔽chromadb遥测版本冲突日志，移除调试打印，系统稳定可用
 2天前
utils
 
fix:修复ingest路径问题、向量入库空embedding异常，增加空文档防护
 8分钟前
.gitignore
 
fix:屏蔽chromadb遥测版本冲突日志，移除调试打印，系统稳定可用
 2天前
README.md
 
update README.md.
 13小时前
__init__.py
 
fix:屏蔽chromadb遥测版本冲突日志，移除调试打印，系统稳定可用
 2天前
app.py
 
feat: 新增rag_chain模块，优化app.py全局复用Retriever，完善端到端RAG命令行问答
 1天前
app_test.py
 
fix:屏蔽chromadb遥测版本冲突日志，移除调试打印，系统稳定可用
 2天前
check_key.py
 
fix:屏蔽chromadb遥测版本冲突日志，移除调试打印，系统稳定可用
 2天前
check_path.py
 
fix:修复ingest路径问题、向量入库空embedding异常，增加空文档防护
 8分钟前
ingest.py
 
fix:屏蔽chromadb遥测版本冲突日志，移除调试打印，系统稳定可用
 2天前
main_cli.py
 
fix:屏蔽chromadb遥测版本冲突日志，移除调试打印，系统稳定可用
 2天前
main_client.py
 
重命名 main_cli.pytype 为 main_client.py
 14小时前
requirements.txt
 
fix:修复chromadb0.6.3 dict对象报错；完成web_gradio网页；锁定运行依赖；新增.gitignore过滤本地产物；…
 8小时前
run_embedder.py
 
fix:屏蔽chromadb遥测版本冲突日志，移除调试打印，系统稳定可用
 2天前
run_ingest.py
 
fix:修复ingest路径问题、向量入库空embedding异常，增加空文档防护
 8分钟前
run_llm.py
 
fix:屏蔽chromadb遥测版本冲突日志，移除调试打印，系统稳定可用
 2天前
run_rag_chain.py
 
更新向量存储、重排器、llm连接器与RAG主链路逻辑
 13小时前
run_retriever.py
 
fix:屏蔽chromadb遥测版本冲突日志，移除调试打印，系统稳定可用
 2天前
run_vector.py
 
fix:屏蔽chromadb遥测版本冲突日志，移除调试打印，系统稳定可用
 2天前
test.py
 
fix:屏蔽chromadb遥测版本冲突日志，移除调试打印，系统稳定可用
 2天前
test_embed.py
 
fix:修复ingest路径问题、向量入库空embedding异常，增加空文档防护
 8分钟前
web_gradio.py
 
fix:修复chromadb0.6.3 dict对象报错；完成web_gradio网页；锁定运行依赖；新增.gitignore过滤本地产物；…
 8小时前
 README
AI
Local‑Knowledge‑RAG‑Chatbot 本地知识库问答系统
命令行交互问答
快速链路测试
Local‑Knowledge‑RAG‑Chatbot 本地知识库问答系统
📖 项目介绍 本项目是一套轻量、完整、可落地的本地知识库 RAG 问答系统，基于 LangChain + Chroma 向量数据库 + 阿里通义 DashScope Embedding/LLM 实现。实现了从文档解析、文本切片、向量化存储、多路混合检索、重排精排、大模型问答、答案溯源的完整端到端 RAG 流程。 项目全程解耦模块化开发，代码规范贴合企业级工程标准，可直接作为简历实战项目、毕设项目、面试口述项目。

🚀 技术栈

核心框架：LangChain（文档处理、链式调用）
向量数据库：Chroma（本地持久化向量库，轻量免部署）
Embedding 模型：阿里 DashScope text‑embedding‑v2（1536维向量）
大模型服务：通义千问系列（OpenAI 兼容接口调用）
稀疏检索：rank‑bm25Okapi + jieba 中文分词，实现中文关键词召回
融合算法：RRF倒数排名融合算法，解决多路检索分数量级不一致问题
重排服务：支持本地BGE离线重排、阿里云DashScope云端重排两套方案，具备API故障降级
开发环境：Python3.10 + Conda 虚拟环境
代码规范：模块化解耦、全局单例复用、参数统一配置、日志优化、异常捕获
🎯 项目核心功能

多格式文档加载：支持 TXT / PDF / DOCX 文档解析读取
智能语义切片：递归字符分割，优先段落、句号分割，最大程度保留语义完整性，支持自定义切片大小与重叠量
文本向量化：调用阿里通义向量模型，生成 1536 维语义向量
向量持久化存储：基于 Chroma 实现本地向量库持久化，断电不丢失数据
多路混合检索：BM25关键词稀疏检索 + 向量稠密语义检索，RRF倒数排名融合，同时兼顾专有名词召回与语义理解
可开关重排模块：支持本地离线重排、云端API重排，接口异常自动降级，保障服务可用
答案溯源能力：返回答案对应的文档来源、相似度距离分数，可溯源校验答案准确性
幻觉抑制机制：通过 System 提示词强约束，无知识库内容时主动兜底拒绝回答，禁止模型编造内容
命令行交互问答：极简交互式问答入口，开箱即用，支持连续提问
工程优化：屏蔽 Chroma 遥测报错、全局检索器单例复用、路径适配、环境隔离、API调用全链路异常捕获兜底
预留扩展：对话记忆模块，方便扩展多轮RAG对话能力
📊 整体架构流程 文档入库流程： 文档加载 → 清洗文本 → 递归语义切片 → Embedding 向量化 → Chroma 向量库持久化存储 → 构建BM25稀疏检索索引

用户问答流程： 用户提问 → BM25关键词检索 + 向量相似度检索 → RRF倒数排名融合 →（可选重排精排）→ 召回知识库上下文 → 拼接约束 Prompt → LLM 生成答案 → 返回回答 + 溯源来源

📁 项目目录结构 Local-Knowledge-RAG-Chatbot ├── chroma_db/ # Chroma向量持久化数据库（git忽略） ├── data/ # 存放本地知识库文档 ├── loaders/ # 多格式文档加载模块 ├── utils/ # 核心工具模块 │ ├── config.py # 全局统一配置文件 │ ├── conversation_memory.py # 对话记忆预留模块 │ ├── text_splitter.py # 语义文本切片器 │ ├── embedder.py # 向量 Embedding 模型封装 │ ├── vector_store.py # Chroma 向量数据库封装 │ ├── hybrid_retriever.py# BM25+向量RRF混合检索器 │ ├── reranker.py # 本地/云端双版本重排器 │ ├── prompt_template.py # Prompt模板管理 │ ├── rag_chain.py # RAG 完整业务链路组装 │ └── llm_connector.py # 大模型调用客户端 ├── vector_db/ ├── venv/ # 虚拟环境（git忽略） ├── ingest.py # 一键文档入库脚本 ├── app.py # Web预留入口 ├── main_cli.py # 命令行交互问答入口 ├── run_embedder.py # 各模块独立测试脚本 ├── run_rag_chain.py # RAG链路测试脚本 ├── run_retriever.py # 检索模块测试脚本 ├── run_vector.py # 向量库测试脚本 ├── test.py ├── .env # 密钥环境配置（忽略上传） ├── .env.example # 环境变量模板，可提交仓库 └── .gitignore # 项目隐私/缓存文件过滤规则

⚙️ 环境部署与启动

环境依赖安装
基于 conda 创建虚拟环境，安装项目全部依赖

conda create -n llm_env python=3.10
conda activate llm_env
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
环境变量配置
根目录复制 .env.example 为 .env，配置阿里通义密钥与模型地址

DASHSCOPE_API_KEY=你的密钥
LLM_MODEL=qwen-turbo
知识库入库
将文档放入 data/ 目录，执行 ingest.py 完成切片、向量化、入库、构建BM25索引

python ingest.py
启动问答系统

命令行交互问答
python main_cli.py

快速链路测试
python run_rag_chain.py

输入问题即可问答，输入 q / exit 退出程序

💡 项目亮点（面试专属口述）

解决大模型幻觉：通过本地知识库检索+强Prompt约束，严格限制模型仅使用已知知识库内容作答
混合检索优化：BM25Okapi中文关键词检索搭配向量检索，RRF倒数排名融合，改善专有名词、专业术语召回效果
双模式重排：本地离线重排 + 云端重排，API异常自动降级，保证服务不会直接不可用
语义切片优化：采用递归分割策略，优先语义符号切割，避免整句语义断裂，提升检索精度
性能优化：检索器全局单例初始化，避免重复加载向量库，减少资源开销
可溯源设计：问答结果附带文档来源与相似度分数，方便问题排查与答案校验
工程健壮性：完善异常捕获、空内容兜底、日志优化、屏蔽第三方库冗余报错
低耦合高扩展：各模块独立封装，可无缝替换本地模型、不同向量库、不同LLM服务
📌 项目总结 本项目完整实现了工业级轻量化 RAG 知识库系统，逻辑闭环、代码规范、可直接部署使用，适合作为 AI 大模型应用、知识库开发、后端工程实战项目。

简介
企业级本地知识库RAG智能问答系统，LangChain+通义千问实现文档解析、向量检索、知识库问答
暂无标签
 README
 1 Stars
 1 Watching
 0 Forks
发行版
暂无发行版， 创建
贡献者 (2)
44s55hx44s55
语言
Python
100.0%
近期动态
hx44s55
8分钟前推送了新的提交到 main 分支，7c967b4...ed0fbbe
hx44s55
8小时前推送了新的提交到 main 分支，1cd3825...7c967b4
hx44s55
12小时前推送了新的提交到 main 分支，ba09d03...1cd3825
hx44s55
13小时前推送了新的提交到 main 分支，a09a3ed...ba09d03
hx44s55
13小时前推送了新的提交到 main 分支，aa44f75...a09a3ed
加载更多 
Gitee - 基于 Git 的代码托管和研发协作平台
北京奥思研工智能科技有限公司版权所有
Git 大全
Git 命令学习
CopyCat 代码克隆检测
APP与插件下载
 
Gitee 封面人物
GVP 项目
Gitee 博客
Gitee 公益计划
Gitee 持续集成
 
OpenAPI
MCP Server
帮助文档
在线自助服务
更新日志
 
关于我们
加入我们
使用条款
意见建议
合作伙伴
 
技术交流QQ群
技术交流QQ群

 
微信服务号
微信服务号

 client@oschina.cn
 企业版在线使用：400-606-0201
专业版私有部署：
赖经理 13058176526
开放原子开源基金会
开放原子开源基金会
合作代码托管平台
违法和不良信息举报中心
违法和不良信息举报中心
京ICP备2025119063号
京公网安备11011502039387号
京公网安备11011502039387号
 简 体 / 繁 體 / English

import os
# 设置 HuggingFace 镜像环境变量（解决网络连接问题）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
import sys
from pathlib import Path
from typing import Optional

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.CustomVLLM import CustomVLLM

# 配置
LAW_DB_DIR = str(project_root / "chroma_db")  # 法条型知识库
CASE_DB_DIR = str(project_root / "chroma_db_case")  # 案例型知识库
JUDGEMENT_DB_DIR = str(project_root / "chroma_db_judgement")  # 判决书型知识库
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
# LLM 服务的端口是 8000，CustomVLLM 默认指向这个地址

# 初始化 LangChain 组件 (全局加载一次)
app = FastAPI()
llm = CustomVLLM() # 连接到你的 vLLM 服务
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

# 初始化多个知识库（法条型 + 案例型 + 判决书型）
law_vectordb: Optional[Chroma] = None
case_vectordb: Optional[Chroma] = None
judgement_vectordb: Optional[Chroma] = None
law_retriever = None
case_retriever = None
judgement_retriever = None

# 加载法条型知识库（如果存在）
if Path(LAW_DB_DIR).exists() and any(Path(LAW_DB_DIR).iterdir()):
    try:
        law_vectordb = Chroma(persist_directory=LAW_DB_DIR, embedding_function=embeddings)
        law_retriever = law_vectordb.as_retriever(search_kwargs={"k": 2})
        print(f"✅ 法条型知识库已加载: {LAW_DB_DIR}")
    except Exception as e:
        print(f"⚠️  法条型知识库加载失败: {e}")

# 加载案例型知识库（如果存在）
if Path(CASE_DB_DIR).exists() and any(Path(CASE_DB_DIR).iterdir()):
    try:
        case_vectordb = Chroma(persist_directory=CASE_DB_DIR, embedding_function=embeddings)
        case_retriever = case_vectordb.as_retriever(search_kwargs={"k": 2})
        print(f"✅ 案例型知识库已加载: {CASE_DB_DIR}")
    except Exception as e:
        print(f"⚠️  案例型知识库加载失败: {e}")

# 加载判决书型知识库（如果存在）
if Path(JUDGEMENT_DB_DIR).exists() and any(Path(JUDGEMENT_DB_DIR).iterdir()):
    try:
        judgement_vectordb = Chroma(persist_directory=JUDGEMENT_DB_DIR, embedding_function=embeddings)
        judgement_retriever = judgement_vectordb.as_retriever(search_kwargs={"k": 1})
        print(f"✅ 判决书型知识库已加载: {JUDGEMENT_DB_DIR}")
    except Exception as e:
        print(f"⚠️  判决书型知识库加载失败: {e}")

# 选择主要的知识库和检索器
# 统计可用的知识库数量
available_dbs = sum([
    law_vectordb is not None,
    case_vectordb is not None,
    judgement_vectordb is not None
])

if available_dbs >= 2:
    # 多个知识库，使用混合检索
    vectordb = law_vectordb or case_vectordb or judgement_vectordb
    retriever = law_retriever or case_retriever or judgement_retriever
    db_names = []
    if law_vectordb:
        db_names.append("法条型")
    if case_vectordb:
        db_names.append("案例型")
    if judgement_vectordb:
        db_names.append("判决书型")
    print(f"📚 混合检索模式：{' + '.join(db_names)}")
elif judgement_vectordb:
    vectordb = judgement_vectordb
    retriever = judgement_retriever
    print("📚 使用判决书型知识库")
elif case_vectordb:
    vectordb = case_vectordb
    retriever = case_retriever
    print("📚 使用案例型知识库")
elif law_vectordb:
    vectordb = law_vectordb
    retriever = law_retriever
    print("📚 使用法条型知识库")
else:
    # 如果都不存在，尝试使用默认路径
    try:
        vectordb = Chroma(persist_directory=LAW_DB_DIR, embedding_function=embeddings)
        retriever = vectordb.as_retriever(search_kwargs={"k": 3})
        print("📚 使用默认知识库路径")
    except Exception as e:
        print(f"❌ 错误: 无法加载任何知识库: {e}")
        vectordb = None
        retriever = None

# 定义 RAG 提示词模板
# 这部分很重要，它指导 LLM 如何使用检索到的知识
RAG_PROMPT_TEMPLATE = """
你是一名专业的法律助手。请根据提供的【上下文】来回答用户的问题。
上下文可能包含法律条文或相关案例。请结合这些信息给出准确、专业的回答。
如果你找不到答案，请诚实地说明你无法找到相关信息，不要编造。

【上下文】：
{context}

用户问题：{question}

请基于上下文中的法律条文和案例，给出详细、准确的法律建议。
"""
RAG_PROMPT = PromptTemplate(
    template=RAG_PROMPT_TEMPLATE, input_variables=["context", "question"]
)

# 封装 RAG 链 (Chain)
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff", # 将所有检索到的文档块填充到上下文
    retriever=retriever,
    chain_type_kwargs={"prompt": RAG_PROMPT}
)

# 定义 API 请求体
class ChatRequest(BaseModel):
    query: str

# 定义 API 接口
@app.post("/api/rag/chat")
async def chat_endpoint(request: ChatRequest):
    """RAG 聊天接口，查询法律知识库并返回结果（支持混合检索）"""
    print(f"📥 收到查询: {request.query}")
    
    if not retriever:
        return {"response": "❌ 错误: 知识库未加载，请先运行 ingest.py 构建知识库"}
    
    # 如果多个知识库都存在，使用混合检索
    available_retrievers = []
    if law_retriever:
        available_retrievers.append(("法条", law_retriever, 2))
    if case_retriever:
        available_retrievers.append(("案例", case_retriever, 1))
    if judgement_retriever:
        available_retrievers.append(("判决书", judgement_retriever, 1))
    
    if len(available_retrievers) >= 2:
        try:
            # 从多个知识库分别检索
            all_docs = []
            retrieval_info = []
            
            for name, ret, k in available_retrievers:
                docs = ret.get_relevant_documents(request.query)
                all_docs.extend(docs[:k])
                retrieval_info.append(f"{name}: {len(docs)}")
            
            # 手动构建上下文
            context = "\n\n".join([doc.page_content for doc in all_docs])
            
            # 使用提示词模板生成回答
            prompt = RAG_PROMPT.format(context=context, question=request.query)
            response = llm.invoke(prompt)
            
            print(f"✅ 混合检索完成（{', '.join(retrieval_info)}）")
            return {"response": response}
        except Exception as e:
            print(f"❌ 混合检索失败: {e}")
            # 降级到单个知识库检索
            pass
    
    # 单个知识库，使用标准 RAG 链
    try:
        result = rag_chain.invoke(request.query)
        return {"response": result['result']}
    except Exception as e:
        return {"response": f"❌ 检索失败: {str(e)}"}

import os
# 设置 HuggingFace 镜像环境变量（解决网络连接问题）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from CustomVLLM import CustomVLLM

# 配置
PERSIST_DIR = "./chroma_db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
# LLM 服务的端口是 8000，CustomVLLM 默认指向这个地址

# 初始化 LangChain 组件 (全局加载一次)
app = FastAPI()
llm = CustomVLLM() # 连接到你的 vLLM 服务
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
vectordb = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
retriever = vectordb.as_retriever(search_kwargs={"k": 2}) # 检索最相关的2个文档块

# 定义 RAG 提示词模板
# 这部分很重要，它指导 LLM 如何使用检索到的知识
RAG_PROMPT_TEMPLATE = """
你是一名专业的法律/金融合同助手。请根据提供的【上下文】来回答用户的问题。
如果你找不到答案，请诚实地说明你无法找到相关信息，不要编造。

【上下文】：
{context}

用户问题：{question}
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
    """RAG 聊天接口，查询法律知识库并返回结果"""
    print(f"Received query: {request.query}")
    
    # 调用 RAG 链
    result = rag_chain.invoke(request.query)
    
    return {"response": result['result']}

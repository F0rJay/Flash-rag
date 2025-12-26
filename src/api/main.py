import os
# 设置 HuggingFace 镜像环境变量（解决网络连接问题）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
import sys
from pathlib import Path
from typing import Optional, List, Iterator
from datetime import datetime

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.CustomVLLM import CustomVLLM
from src.core.query_rewriter import QueryRewriter, create_query_rewriter
from src.core.reranker import Reranker, create_reranker
from src.api.monitoring import get_metrics_collector
import time

# 配置
LAW_DB_DIR = str(project_root / "chroma_db")  # 法条型知识库
CASE_DB_DIR = str(project_root / "chroma_db_case")  # 案例型知识库
JUDGEMENT_DB_DIR = str(project_root / "chroma_db_judgement")  # 判决书型知识库
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
# LLM 服务的端口是 8000，CustomVLLM 默认指向这个地址
VLLM_URL = os.getenv("VLLM_URL", "http://localhost:8000")

# 初始化 LangChain 组件 (全局加载一次)
app = FastAPI()
llm = CustomVLLM() # 连接到你的 vLLM 服务
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

# 初始化监控指标收集器
metrics_collector = get_metrics_collector(vllm_url=VLLM_URL)

# 初始化 RAG 优化组件
query_rewriter = None
reranker = None

# 初始化 Query Rewriter（查询改写）
try:
    query_rewriter = create_query_rewriter(llm=llm)
    print("✅ Query Rewriter 已初始化")
except Exception as e:
    print(f"⚠️  Query Rewriter 初始化失败: {e}，将跳过查询改写步骤")

# 初始化 Reranker（重排序）
try:
    reranker = create_reranker(model_name="BAAI/bge-reranker-base")
    print("✅ Reranker 已初始化")
except Exception as e:
    print(f"⚠️  Reranker 初始化失败: {e}，将跳过重排序步骤")

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
    temperature: float = 0.1
    max_tokens: int = 1024
    stream: bool = False  # 是否启用流式输出

# 定义 API 接口
@app.post("/api/rag/chat")
async def chat_endpoint(request: ChatRequest):
    """
    RAG 聊天接口，完整的检索增强生成流程：
    1. Query Rewrite: 改写用户问题为专业检索关键词
    2. Retrieve: 向量检索获取 Top 50 文档
    3. Rerank: 使用 Cross-Encoder 重排序到 Top 5
    4. Generate: LLM 生成最终答案
    """
    start_time = time.time()
    print(f"📥 收到查询: {request.query}")
    
    if not retriever:
        latency = time.time() - start_time
        metrics_collector.record_request(latency, success=False)
        return {"response": "❌ 错误: 知识库未加载，请先运行 ingest.py 构建知识库"}
    
    # === 步骤 1: Query Rewrite (查询改写) ===
    search_query = request.query
    if query_rewriter:
        try:
            search_query = query_rewriter.rewrite(request.query)
            print(f"📝 查询已改写: '{request.query}' -> '{search_query}'")
        except Exception as e:
            print(f"⚠️  查询改写失败，使用原查询: {e}")
            search_query = request.query
    else:
        search_query = request.query
    
    # === 步骤 2: Retrieve (向量检索) ===
    # 如果多个知识库都存在，使用混合检索
    available_retrievers = []
    if law_retriever:
        available_retrievers.append(("法条", law_retriever, 50))  # 先取 Top 50
    if case_retriever:
        available_retrievers.append(("案例", case_retriever, 50))
    if judgement_retriever:
        available_retrievers.append(("判决书", judgement_retriever, 50))
    
    all_docs = []
    retrieval_info = []
    
    if len(available_retrievers) >= 2:
        # 多知识库混合检索
        try:
            for name, ret, k in available_retrievers:
                docs = ret.get_relevant_documents(search_query)
                all_docs.extend(docs[:k])
                retrieval_info.append(f"{name}: {len(docs)}")
            print(f"🔍 向量检索完成（{', '.join(retrieval_info)}），共 {len(all_docs)} 个文档")
        except Exception as e:
            print(f"❌ 混合检索失败: {e}")
            return {"response": f"❌ 检索失败: {str(e)}"}
    else:
        # 单个知识库检索
        try:
            if available_retrievers:
                name, ret, k = available_retrievers[0]
                docs = ret.get_relevant_documents(search_query)
                all_docs = docs[:k]
                retrieval_info.append(f"{name}: {len(docs)}")
                print(f"🔍 向量检索完成，共 {len(all_docs)} 个文档")
            else:
                # 降级到标准 RAG 链
                try:
                    result = rag_chain.invoke(request.query)
                    return {"response": result['result']}
                except Exception as e:
                    return {"response": f"❌ 检索失败: {str(e)}"}
        except Exception as e:
            print(f"❌ 检索失败: {e}")
            return {"response": f"❌ 检索失败: {str(e)}"}
    
    if not all_docs:
        return {"response": "❌ 未检索到相关文档，请尝试其他问题"}
    
    # === 步骤 3: Rerank (重排序) ===
    # 将文档转换为字符串列表用于重排序
    doc_contents = [doc.page_content for doc in all_docs]
    doc_metadata = [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in all_docs]
    
    if reranker and len(doc_contents) > 5:
        try:
            # 使用重排序器对文档进行精细排序
            reranked_docs = reranker.rerank_with_metadata(
                query=request.query,  # 使用原始查询进行重排序
                documents_with_metadata=doc_metadata,
                top_k=5
            )
            print(f"🎯 重排序完成，从 {len(doc_contents)} 个文档中选出 Top 5")
            # 提取重排序后的文档内容
            final_docs = [doc['page_content'] for doc in reranked_docs]
        except Exception as e:
            print(f"⚠️  重排序失败，使用原始检索结果: {e}")
            # 重排序失败，使用原始 Top 5
            final_docs = doc_contents[:5]
    else:
        # 如果没有重排序器或文档数量较少，直接取 Top 5
        final_docs = doc_contents[:5]
        if reranker:
            print(f"ℹ️  文档数量较少（{len(doc_contents)}），跳过重排序")
    
    # === 步骤 4: Generate (生成答案) ===
    try:
        # 构建上下文
        context = "\n\n".join([f"[文档 {i+1}]\n{doc}" for i, doc in enumerate(final_docs)])
        
        # 使用提示词模板生成回答
        prompt = RAG_PROMPT.format(context=context, question=request.query)
        
        # 如果启用流式输出
        if request.stream:
            return StreamingResponse(
                _stream_response(
                    llm=llm,
                    prompt=prompt,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                    sources=final_docs,
                    start_time=start_time
                ),
                media_type="text/event-stream"
            )
        else:
            # 非流式输出
            response = llm.invoke(prompt)
            
            print(f"✅ RAG 流程完成: 改写 → 检索({len(all_docs)}) → 重排序({len(final_docs)}) → 生成")
            return {
                "response": response,
                "sources": [
                    {"content": doc[:200] + "..." if len(doc) > 200 else doc, "index": i+1}
                    for i, doc in enumerate(final_docs)
                ]
            }
    except Exception as e:
        print(f"❌ 生成失败: {e}")
        if request.stream:
            # 流式输出错误
            def error_stream():
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
            return StreamingResponse(error_stream(), media_type="text/event-stream")
        else:
            return {"response": f"❌ 生成失败: {str(e)}", "sources": []}


def _stream_response(
    llm: CustomVLLM,
    prompt: str,
    temperature: float = 0.1,
    max_tokens: int = 1024,
    sources: List[str] = None,
    start_time: float = None
) -> Iterator[str]:
    """
    流式响应生成器
    
    Args:
        llm: CustomVLLM 实例
        prompt: 提示词
        temperature: 温度参数
        max_tokens: 最大 token 数
        sources: 检索到的文档列表
        start_time: 请求开始时间（用于延迟统计）
        
    Yields:
        str: SSE 格式的数据流
    """
    # 发送开始信号
    yield f"data: {json.dumps({'type': 'start'})}\n\n"
    
    # 流式生成
    full_response = ""
    success = True
    try:
        for chunk in llm.stream(prompt, temperature=temperature, max_tokens=max_tokens):
            full_response += chunk
            # 发送文本块
            yield f"data: {json.dumps({'type': 'chunk', 'text': chunk})}\n\n"
        
        # 发送结束信号和来源信息
        yield f"data: {json.dumps({'type': 'done', 'sources': sources or []})}\n\n"
    except Exception as e:
        success = False
        yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
    finally:
        # 记录延迟指标（流式输出）
        if start_time is not None:
            latency = time.time() - start_time
            metrics_collector.record_request(latency, success=success)


# 健康检查端点（增强版）
@app.get("/health")
async def health_check():
    """
    增强的健康检查端点
    检查：vLLM 连接、知识库状态、服务可用性
    """
    health_status = {
        "status": "healthy",
        "service": "LegalFlash-RAG API",
        "timestamp": datetime.now().isoformat(),
        "checks": {}
    }
    
    # 检查 vLLM 服务
    vllm_health = metrics_collector.check_vllm_health()
    health_status["checks"]["vllm"] = vllm_health
    
    # 检查知识库状态
    knowledge_bases = {
        "law": Path(LAW_DB_DIR).exists() and any(Path(LAW_DB_DIR).iterdir()),
        "case": Path(CASE_DB_DIR).exists() and any(Path(CASE_DB_DIR).iterdir()),
        "judgement": Path(JUDGEMENT_DB_DIR).exists() and any(Path(JUDGEMENT_DB_DIR).iterdir())
    }
    health_status["checks"]["knowledge_bases"] = knowledge_bases
    health_status["checks"]["available_retrievers"] = sum([
        law_retriever is not None,
        case_retriever is not None,
        judgement_retriever is not None
    ])
    
    # 检查 RAG 组件
    health_status["checks"]["components"] = {
        "query_rewriter": query_rewriter is not None,
        "reranker": reranker is not None,
        "embeddings": embeddings is not None,
        "llm": llm is not None
    }
    
    # 如果 vLLM 不可用，标记为不健康
    if vllm_health["status"] != "healthy":
        health_status["status"] = "degraded"
    
    # 如果没有可用的知识库，标记为不健康
    if health_status["checks"]["available_retrievers"] == 0:
        health_status["status"] = "unhealthy"
    
    return health_status


# 监控指标端点
@app.get("/metrics")
async def get_metrics():
    """
    获取系统监控指标
    包括：GPU 使用率、延迟统计、吞吐量、CPU/内存使用情况
    """
    return metrics_collector.get_all_metrics()


# 监控指标端点（Prometheus 格式，可选）
@app.get("/metrics/prometheus")
async def get_prometheus_metrics():
    """
    获取 Prometheus 格式的监控指标
    """
    metrics = metrics_collector.get_all_metrics()
    
    # 转换为 Prometheus 格式
    prometheus_lines = []
    
    # 请求统计
    prometheus_lines.append(f'legalflash_rag_requests_total {metrics["requests"]["total"]}')
    prometheus_lines.append(f'legalflash_rag_requests_errors_total {metrics["requests"]["errors"]}')
    prometheus_lines.append(f'legalflash_rag_requests_success_rate {metrics["requests"]["success_rate"]}')
    
    # 延迟统计
    latency = metrics["latency"]
    prometheus_lines.append(f'legalflash_rag_latency_avg_seconds {latency["avg"]}')
    prometheus_lines.append(f'legalflash_rag_latency_p95_seconds {latency["p95"]}')
    prometheus_lines.append(f'legalflash_rag_latency_p99_seconds {latency["p99"]}')
    
    # 吞吐量
    throughput = metrics["throughput"]
    prometheus_lines.append(f'legalflash_rag_throughput_rps_1min {throughput["requests_per_second_1min"]}')
    
    # GPU 指标
    for gpu in metrics["gpu"]:
        idx = gpu["index"]
        prometheus_lines.append(f'legalflash_rag_gpu_memory_used_gb{{gpu="{idx}"}} {gpu["memory"]["used_gb"]}')
        prometheus_lines.append(f'legalflash_rag_gpu_utilization_percent{{gpu="{idx}"}} {gpu["utilization_percent"]}')
    
    return "\n".join(prometheus_lines)

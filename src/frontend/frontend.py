import streamlit as st
import requests
import json

# === 配置 ===
BACKEND_URL = "http://localhost:8080/api/rag/chat"
st.set_page_config(page_title="LegalFlash-RAG 法律助手", page_icon="⚖️", layout="wide")

# === 界面标题 ===
st.title("⚖️ LegalFlash-RAG 法律智能助手")
st.caption("🚀 Powered by Llama 3 (LoRA) + vLLM + LangChain + RAG")

# === 侧边栏：项目介绍和参数设置 ===
with st.sidebar:
    st.header("⚙️ 参数设置")
    
    # 温度参数
    temperature = st.slider(
        "Temperature (温度)",
        min_value=0.0,
        max_value=2.0,
        value=0.1,
        step=0.1,
        help="控制生成的随机性。值越大，输出越随机；值越小，输出越确定。"
    )
    
    # 最大 token 数
    max_tokens = st.slider(
        "Max Tokens (最大长度)",
        min_value=100,
        max_value=2048,
        value=1024,
        step=100,
        help="生成答案的最大 token 数。"
    )
    
    # 流式输出开关
    stream_enabled = st.checkbox(
        "启用流式输出",
        value=True,
        help="启用后，答案会逐字显示，体验更流畅。"
    )
    
    st.divider()
    
    st.header("📚 关于项目")
    st.markdown("""
    这是一个基于 **RAG (检索增强生成)** 的垂直领域问答系统。
    
    **核心技术栈：**
    - 🧠 **模型**: Llama 3 (8B) + LoRA 微调
    - ⚡ **推理**: vLLM 高性能引擎
    - 🔗 **后端**: FastAPI + LangChain
    - 🗄️ **知识库**: ChromaDB (法律条文、案例、判决书)
    - 🔄 **Query Rewrite**: 智能查询改写
    - 🎯 **Rerank**: Cross-Encoder 重排序
    """)
    
    if st.button("🗑️ 清除对话历史"):
        st.session_state.messages = []
        st.rerun()

# === 初始化对话历史 ===
if "messages" not in st.session_state:
    st.session_state.messages = []
if "sources" not in st.session_state:
    st.session_state.sources = []

# === 展示历史消息 ===
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # 如果是助手消息，显示来源
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("📄 查看参考来源", expanded=False):
                for i, source in enumerate(message["sources"], 1):
                    st.markdown(f"**来源 {i}:**")
                    st.text(source[:500] + "..." if len(source) > 500 else source)
                    st.divider()

# === 处理用户输入 ===
if prompt := st.chat_input("请输入关于合同违约、借款期限等法律问题..."):
    # 1. 展示用户问题
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. 调用后端 API
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        sources_placeholder = st.empty()
        
        try:
            if stream_enabled:
                # === 流式输出模式 ===
                message_placeholder.markdown("🤔 正在检索法律条文并思考...")
                
                # 准备请求参数
                payload = {
                    "query": prompt,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stream": True
                }
                
                # 发送流式请求
                response = requests.post(
                    BACKEND_URL,
                    json=payload,
                    stream=True,
                    timeout=120
                )
                
                if response.status_code == 200:
                    full_response = ""
                    sources = []
                    
                    # 处理 SSE 流式响应
                    for line in response.iter_lines():
                        if line:
                            line_text = line.decode('utf-8')
                            if line_text.startswith('data: '):
                                data_str = line_text[6:]
                                try:
                                    data = json.loads(data_str)
                                    
                                    if data.get('type') == 'start':
                                        message_placeholder.markdown("")
                                        full_response = ""
                                    elif data.get('type') == 'chunk':
                                        chunk = data.get('text', '')
                                        full_response += chunk
                                        message_placeholder.markdown(full_response + "▌")
                                    elif data.get('type') == 'done':
                                        message_placeholder.markdown(full_response)
                                        sources = data.get('sources', [])
                                    elif data.get('type') == 'error':
                                        message_placeholder.error(f"错误: {data.get('error')}")
                                        break
                                except json.JSONDecodeError:
                                    continue
                    
                    # 保存完整回复和来源
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_response,
                        "sources": sources
                    })
                    
                    # 显示来源
                    if sources:
                        with sources_placeholder.expander("📄 查看参考来源", expanded=False):
                            for i, source in enumerate(sources, 1):
                                source_content = source.get('content', source) if isinstance(source, dict) else source
                                st.markdown(f"**来源 {i}:**")
                                st.text(source_content[:500] + "..." if len(source_content) > 500 else source_content)
                                st.divider()
                else:
                    message_placeholder.error(f"后端报错: {response.status_code}")
            else:
                # === 非流式输出模式 ===
                message_placeholder.markdown("🤔 正在检索法律条文并思考...")
                
                # 发送请求给 FastAPI
                response = requests.post(
                    BACKEND_URL,
                    json={
                        "query": prompt,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "stream": False
                    },
                    timeout=120
                )
                
                if response.status_code == 200:
                    data = response.json()
                    ai_response = data.get("response", "")
                    sources = data.get("sources", [])
                    
                    message_placeholder.markdown(ai_response)
                    
                    # 保存 AI 回复到历史
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": ai_response,
                        "sources": sources
                    })
                    
                    # 显示来源
                    if sources:
                        with sources_placeholder.expander("📄 查看参考来源", expanded=False):
                            for i, source in enumerate(sources, 1):
                                source_content = source.get('content', source) if isinstance(source, dict) else source
                                st.markdown(f"**来源 {i}:**")
                                st.text(source_content[:500] + "..." if len(source_content) > 500 else source_content)
                                st.divider()
                else:
                    message_placeholder.error(f"后端报错: {response.status_code}")
                    
        except requests.exceptions.Timeout:
            message_placeholder.error("⏱️ 请求超时，请稍后重试。")
        except requests.exceptions.ConnectionError:
            message_placeholder.error("🔌 连接失败，请检查 FastAPI 服务是否已启动。")
        except Exception as e:
            message_placeholder.error(f"❌ 错误: {str(e)}")

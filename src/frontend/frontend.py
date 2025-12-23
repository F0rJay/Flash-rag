import streamlit as st
import requests
import json

# === 配置 ===
BACKEND_URL = "http://localhost:8080/api/rag/chat"
st.set_page_config(page_title="Flash-RAG 法律助手", page_icon="⚖️")

# === 界面标题 ===
st.title("⚖️ Flash-RAG 法律智能助手")
st.caption("🚀 Powered by Llama 3 (LoRA) + vLLM + LangChain")

# === 侧边栏：项目介绍 ===
with st.sidebar:
    st.header("关于项目")
    st.markdown("""
    这是一个基于 **RAG (检索增强生成)** 的垂直领域问答系统。
    
    **核心技术栈：**
    - 🧠 **模型**: Llama 3 (8B) + LoRA 微调
    - ⚡ **推理**: vLLM 高性能引擎
    - 🔗 **后端**: FastAPI + LangChain
    - 🗄️ **知识库**: ChromaDB (法律合同数据)
    """)
    if st.button("清除对话历史"):
        st.session_state.messages = []

# === 初始化对话历史 ===
if "messages" not in st.session_state:
    st.session_state.messages = []

# === 展示历史消息 ===
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# === 处理用户输入 ===
if prompt := st.chat_input("请输入关于合同违约、借款期限等法律问题..."):
    # 1. 展示用户问题
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. 调用后端 API
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("🤔 正在检索法律条文并思考...")
        
        try:
            # 发送请求给 FastAPI
            response = requests.post(
                BACKEND_URL, 
                json={"query": prompt},
                timeout=60 # 防止超时
            )
            
            if response.status_code == 200:
                ai_response = response.json()["response"]
                message_placeholder.markdown(ai_response)
                # 保存 AI 回复到历史
                st.session_state.messages.append({"role": "assistant", "content": ai_response})
            else:
                message_placeholder.error(f"后端报错: {response.status_code}")
                
        except Exception as e:
            message_placeholder.error(f"连接失败: {str(e)}。请检查 FastAPI 是否已启动。")
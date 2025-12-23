import os
from pathlib import Path
# 设置 HuggingFace 镜像环境变量（解决网络连接问题）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# 获取项目根目录
project_root = Path(__file__).parent.parent.parent

# 定义向量库路径
PERSIST_DIR = str(project_root / "chroma_db")
# 定义用于嵌入的开源模型（需本地安装 sentence-transformers）
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2" # 这是一个常用的快速模型

def run_ingestion():
    # 1. 加载文档 (Load Documents)
    docs_path = project_root / "docs" / "legal_docs.txt"
    loader = TextLoader(str(docs_path))
    documents = loader.load()
    print(f"Loaded {len(documents)} document(s).")
    
    # 2. 文档切分 (Text Splitting)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,        # 每个块最大 500 字符
        chunk_overlap=50       # 块之间重叠 50 字符，保持上下文
    )
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks.")

    # 3. 创建嵌入模型 (Create Embeddings)
    # 这将负责将文本转换为高维向量
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    
    # 4. 存储到向量数据库 (Store in VectorDB)
    # 这是创建 RAG 知识库的核心步骤
    vectordb = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )
    # 注意：新版本的 Chroma 在使用 persist_directory 时会自动持久化，无需手动调用 persist()
    print(f"Ingestion complete. Knowledge base saved to {PERSIST_DIR}")

if __name__ == "__main__":
    # 确保 docs 文件夹和文件存在
    if not os.path.exists("docs/legal_docs.txt"):
        print("ERROR: Please create docs/legal_docs.txt first!")
    else:
        run_ingestion()
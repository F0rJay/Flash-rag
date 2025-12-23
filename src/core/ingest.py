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

# 定义向量库路径（支持多个知识库）
DEFAULT_PERSIST_DIR = str(project_root / "chroma_db")
# 定义用于嵌入的开源模型（需本地安装 sentence-transformers）
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2" # 这是一个常用的快速模型

def run_ingestion(docs_path=None, chunk_size=500, chunk_overlap=50, persist_dir=None, knowledge_type="law"):
    """
    运行文档向量化处理
    
    Args:
        docs_path: 文档路径（默认: data/docs/legal_docs.txt）
        chunk_size: 文档块大小（默认: 500 字符）
        chunk_overlap: 块之间重叠大小（默认: 50 字符）
        persist_dir: 向量库保存路径（默认根据 knowledge_type 自动生成）
        knowledge_type: 知识库类型 ("law"=法条型, "case"=案例型, "judgement"=判决书型, 默认: "law")
    """
    # 1. 加载文档 (Load Documents)
    if docs_path is None:
        docs_path = project_root / "data" / "docs" / "legal_docs.txt"
    else:
        docs_path = Path(docs_path)
        if not docs_path.is_absolute():
            docs_path = project_root / docs_path
    
    if not docs_path.exists():
        print(f"❌ 错误: 文档文件不存在: {docs_path}")
        print(f"💡 提示: 请先运行 'python scripts/prepare_rag_knowledge.py' 准备知识库")
        return None
    
    print(f"📂 加载文档: {docs_path}")
    loader = TextLoader(str(docs_path), encoding='utf-8')
    documents = loader.load()
    print(f"✅ 加载了 {len(documents)} 个文档")
    
    # 2. 文档切分 (Text Splitting)
    # 对于法律条文，适当增大 chunk_size 以保持完整性
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,        # 每个块最大字符数
        chunk_overlap=chunk_overlap,  # 块之间重叠字符数，保持上下文
        separators=["\n\n", "\n", "。", "；", "，", " ", ""]  # 优先按段落分割
    )
    texts = text_splitter.split_documents(documents)
    print(f"✅ 切分为 {len(texts)} 个文档块")

    # 3. 创建嵌入模型 (Create Embeddings)
    # 这将负责将文本转换为高维向量
    print(f"🔄 初始化嵌入模型: {EMBEDDING_MODEL_NAME}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    
    # 4. 存储到向量数据库 (Store in VectorDB)
    # 这是创建 RAG 知识库的核心步骤
    
    # 确定向量库保存路径
    if persist_dir is None:
        if knowledge_type == "case":
            persist_dir = str(project_root / "chroma_db_case")
        elif knowledge_type == "judgement":
            persist_dir = str(project_root / "chroma_db_judgement")
        else:
            persist_dir = DEFAULT_PERSIST_DIR
    else:
        persist_dir = str(Path(persist_dir).resolve())
    
    print(f"💾 构建向量数据库...")
    print(f"📁 保存路径: {persist_dir}")
    vectordb = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    # 注意：新版本的 Chroma 在使用 persist_directory 时会自动持久化，无需手动调用 persist()
    print(f"✅ 向量化完成！知识库已保存到: {persist_dir}")
    print(f"📊 统计: {len(texts)} 个文档块已向量化")
    return vectordb

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='文档向量化处理（构建 RAG 知识库）')
    parser.add_argument('--docs-path', type=str, default=None,
                       help='文档文件路径（默认: data/docs/legal_docs.txt）')
    parser.add_argument('--chunk-size', type=int, default=500,
                       help='文档块大小（默认: 500 字符）')
    parser.add_argument('--chunk-overlap', type=int, default=50,
                       help='块之间重叠大小（默认: 50 字符）')
    parser.add_argument('--persist-dir', type=str, default=None,
                       help='向量库保存路径（默认根据知识库类型自动生成）')
    parser.add_argument('--knowledge-type', type=str, choices=['law', 'case', 'judgement'], default='law',
                       help='知识库类型: law=法条型, case=案例型, judgement=判决书型（默认: law）')
    
    args = parser.parse_args()
    
    # 根据知识库类型设置默认文档路径
    if args.docs_path is None:
        if args.knowledge_type == "case":
            args.docs_path = project_root / "data" / "docs" / "case_docs.txt"
        elif args.knowledge_type == "judgement":
            args.docs_path = project_root / "data" / "docs" / "judgement_docs.txt"
        else:
            args.docs_path = project_root / "data" / "docs" / "legal_docs.txt"
    
    run_ingestion(
        docs_path=args.docs_path,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        persist_dir=args.persist_dir,
        knowledge_type=args.knowledge_type
    )
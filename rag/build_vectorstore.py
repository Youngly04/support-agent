from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


BASE_DIR = Path(__file__).resolve().parent.parent
KB_PATH = BASE_DIR / "knowledge_base"
DB_PATH = BASE_DIR / "vectorstore" / "faiss_index"


def load_documents():
    loader = DirectoryLoader(
        str(KB_PATH),
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True
    )
    return loader.load()


def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " "]
    )
    return splitter.split_documents(docs)


def build_vectorstore():
    print("开始加载文档...")
    docs = load_documents()
    print(f"原始文档数: {len(docs)}")

    print("开始切分文档...")
    split_docs = split_documents(docs)
    print(f"切分后 chunk 数: {len(split_docs)}")

    print("开始加载 embedding 模型...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    print("开始构建 FAISS 向量库...")
    vectorstore = FAISS.from_documents(split_docs, embeddings)

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(DB_PATH))

    print(f"向量库已保存到: {DB_PATH}")


if __name__ == "__main__":
    build_vectorstore()
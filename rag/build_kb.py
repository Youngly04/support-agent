from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


BASE_DIR = Path(__file__).resolve().parent.parent
KB_PATH = BASE_DIR / "knowledge_base"


def load_documents():
    loader = DirectoryLoader(
        str(KB_PATH),
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True
    )
    docs = loader.load()
    return docs


def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " "]
    )
    split_docs = splitter.split_documents(docs)
    return split_docs


if __name__ == "__main__":
    print("知识库目录：", KB_PATH)

    docs = load_documents()
    print(f"原始文档数: {len(docs)}")

    split_docs = split_documents(docs)
    print(f"切分后 chunk 数: {len(split_docs)}")

    print("\n===== 前 3 个 chunk 预览 =====")
    for i, doc in enumerate(split_docs[:3]):
        print(f"\n--- chunk {i+1} ---")
        print("来源:", doc.metadata.get("source", ""))
        print(doc.page_content)
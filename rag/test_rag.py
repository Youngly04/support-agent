from pathlib import Path

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, SystemMessage

from rag.chat_model import MyLocalChatModel


BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "vectorstore" / "faiss_index"


def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    vectorstore = FAISS.load_local(
        str(DB_PATH),
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore


def build_context(docs):
    context_parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "")
        content = doc.page_content
        context_parts.append(f"[参考资料{i}] 来源: {source}\n{content}")
    return "\n\n".join(context_parts)


if __name__ == "__main__":
    query = "订单一直没发货怎么办？"

    # 1. 检索
    vectorstore = load_vectorstore()
    docs = vectorstore.similarity_search(query, k=2)

    context = build_context(docs)

    # 2. 调模型
    model = MyLocalChatModel()

    messages = [
        SystemMessage(
            content=(
                "你是一个电商售后客服助手。"
                "你的任务是严格依据参考资料回答。"
                "如果参考资料中已经给出处理步骤，必须尽量完整保留这些步骤。"
                "不要只做笼统概括，不要遗漏关键操作。"
                "如果资料不足，就明确说明“根据当前资料暂时无法确认”。"
            )
        ),
        HumanMessage(
            content=(
                f"参考资料如下：\n{context}\n\n"
                f"用户问题：{query}\n\n"
                "请严格根据参考资料回答，并按照下面格式输出：\n"
                "1. 先检查什么\n"
                "2. 再查看什么\n"
                "3. 如果仍未解决，怎么办\n\n"
                "要求：\n"
                "1. 尽量覆盖参考资料中的关键步骤；\n"
                "2. 不要编造资料里没有的规则；\n"
                "3. 回答要简洁、明确、像客服指引。"
            )
        )
    ]

    result = model.invoke(messages)

    print("\n===== RAG 回答结果 =====")
    print(result.content)
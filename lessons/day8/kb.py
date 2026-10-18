"""Day8：本地知识库。把 knowledge/ 里的文档切块、向量化、存起来，再按意思搜索。"""

from functools import lru_cache

from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from common.llm import ROOT, get_embeddings

KB_DIR = ROOT / "knowledge"


def load_docs() -> list[Document]:
    """第 1 步：读文档。每个 .md 文件变成一个 Document，记下它来自哪个文件。"""
    docs = []
    for path in KB_DIR.glob("*.md"):
        text = path.read_text(encoding="utf-8")
        docs.append(Document(page_content=text, metadata={"source": path.name}))
    return docs


@lru_cache  # 只建一次：向量化要花时间，不用每次搜索都重建
def build_store() -> InMemoryVectorStore:
    """第 2 步：切块。第 3 步：向量化并存进向量库。"""
    splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=20)
    chunks = splitter.split_documents(load_docs())
    store = InMemoryVectorStore(get_embeddings())
    store.add_documents(chunks)
    print(f"[知识库] 读了 {len(load_docs())} 个文档，切成 {len(chunks)} 块")
    return store


@tool
def search_kb(query: str) -> str:
    """查询星辰科技公司内部知识库：请假、报销等规章制度，以及公司产品信息。"""
    # 第 4 步：按意思搜索，找出最相关的 3 块
    results = build_store().similarity_search(query, k=3)
    texts = []
    for doc in results:
        texts.append(f"【来源：{doc.metadata['source']}】\n{doc.page_content}")
    return "\n\n".join(texts)

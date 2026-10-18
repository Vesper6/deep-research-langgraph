"""两个检索工具：联网搜索、本地知识库。每条结果都带来源，写报告时用来标引用。"""

from functools import lru_cache

from ddgs import DDGS
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import MarkdownHeaderTextSplitter

from common.llm import ROOT, get_embeddings

KB_DIR = ROOT / "knowledge"


# ===== 工具 1：联网搜索 =====
def web_search(query: str, k: int = 3) -> list[dict]:
    """联网搜索，返回 [{"source": 网址, "text": 摘要}, ...]。"""
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=k) or []
    except Exception as e:
        print(f"[联网搜索] 失败：{e}")
        return []
    hits = []
    for r in results:
        hits.append({"source": r["href"], "text": r["body"]})
    return hits


# ===== 工具 2：本地知识库 =====
# Day11 评测后改进：按 Markdown 标题切块，每块开头带上“文档 - 小节”标题，
# 否则“病假”那块不知道自己属于“星辰科技 请假制度”，长问题搜不到它。
@lru_cache
def build_store() -> InMemoryVectorStore:
    splitter = MarkdownHeaderTextSplitter([("#", "doc"), ("##", "section")])
    chunks = []
    for path in KB_DIR.glob("*.md"):
        for c in splitter.split_text(path.read_text(encoding="utf-8")):
            # TODO：在 c.page_content 前面加上标题，变成 “文档 - 小节\n原来的内容”
            c.page_content=f"{c.metadata['doc']} - {c.metadata['section']}\n{c.page_content}"
            c.metadata["source"] = path.name
            chunks.append(c)
    store = InMemoryVectorStore(get_embeddings())
    store.add_documents(chunks)
    return store


def search_kb(query: str, k: int = 3) -> list[dict]:
    """查本地知识库，返回格式和 web_search 一样。"""
    results = build_store().similarity_search(query, k=k)
    hits = []
    for doc in results:
        hits.append({"source":doc.metadata["source"],"text":doc.page_content})
    return hits


if __name__ == "__main__":
    # 自检：python -m deep_research.tools
    print(web_search("腾讯 主要业务"))
    print(search_kb("年假有几天"))

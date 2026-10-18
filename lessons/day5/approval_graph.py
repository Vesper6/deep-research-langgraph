"""Day5：写邮件 -> 人工审核 -> 发送 / 取消。"""

from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from common.llm import get_llm


class State(TypedDict):
    topic: str     # 邮件主题
    draft: str     # AI 写的草稿
    decision: str  # 人的决定


def write_draft(state: State) -> dict:
    reply = get_llm().invoke(f"写一封 50 字以内的邮件，主题：{state['topic']}")
    return {"draft": reply.content}


def human_review(state: State) -> dict:
    # 程序在这里暂停，把草稿交给人看；人回复后，回复的内容就是 interrupt 的返回值
    decision = interrupt({"请审核草稿": state["draft"]})
    return {"decision": decision}


def send(state: State) -> dict:
    print("📧 邮件已发送！")
    return {}


def cancel(state: State) -> dict:
    print("🚫 已取消发送。")
    return {}


def route(state: State) -> str:
    return "send" if state["decision"] == "同意" else "cancel"


def build_graph():
    graph = StateGraph(State)
    graph.add_node("write_draft", write_draft)
    graph.add_node("human_review", human_review)
    graph.add_node("send", send)
    graph.add_node("cancel", cancel)

    graph.add_edge(START, "write_draft")
    graph.add_edge("write_draft", "human_review")
    graph.add_conditional_edges("human_review", route, {"send": "send", "cancel": "cancel"})
    graph.add_edge("send", END)
    graph.add_edge("cancel", END)
    # 暂停要靠 checkpointer 存快照，所以 interrupt 必须配 checkpointer
    return graph.compile(checkpointer=InMemorySaver())

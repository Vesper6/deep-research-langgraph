"""DeepResearch 主流程：规划 → 人工审核 → 并行检索 → 反思补搜 → 写报告。"""

import operator
from typing import Annotated, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send, interrupt
from pydantic import BaseModel, Field

from common.llm import get_llm
from deep_research.tools import search_kb, web_search

MAX_ROUNDS = 2  # 最多检索几轮：第 1 轮 + 补搜 1 轮，防止无限循环


# ===== 第 1 段：模型要填的“表格” + 状态 =====
class Plan(BaseModel):
    questions: list[str] = Field(description="拆出来的子问题，每个都能单独搜索")


class Pick(BaseModel):
    tool: Literal["web", "kb"] = Field(description="web：联网搜公开信息；kb：查星辰科技公司内部的制度和产品")


class Reflection(BaseModel):
    enough: bool = Field(description="资料是否足够写报告")
    questions: list[str] = Field(description="不够时还要补搜的问题，最多 2 个；够了就留空")


class State(TypedDict):
    topic: str
    questions: list[str]                       # 这一轮要搜的子问题
    feedback: str                              # 人工审核的意见
    hits: Annotated[list[dict], operator.add]  # 所有搜到的资料，每条带来源
    rounds: int                                # 已经搜了几轮
    enough: bool                               # 反思后：资料够不够
    report: str


def format_hits(hits: list[dict]) -> str:
    """把资料变成带编号的文字：[1] xxx  [2] xxx"""
    lines = []
    for i, h in enumerate(hits, 1):
        lines.append(f"[{i}] {h['text']}")
    return "\n".join(lines)


def unique(hits: list[dict]) -> list[dict]:
    """去掉重复的资料：一模一样的只留第一条。"""
    result = []
    for h in hits:
        # TODO：如果 h 还不在 result 里，就把它加进去
        if h not in result:
            result.append(h)
    return result


# ===== 第 2 段：规划 + 人工审核 =====
def plan(state: State) -> dict:
    prompt = f"把这个研究主题拆成 3 个子问题：{state['topic']}"
    if state["feedback"]:
        prompt += f"\n用户对上一版的修改意见：{state['feedback']}"
    result = get_llm().with_structured_output(Plan).invoke(prompt)
    return {"questions": result.questions}


def human_review(state: State) -> dict:
    decision = interrupt({"请审核子问题": state["questions"]})
    return {"feedback": decision}


def after_review(state: State):
    if state["feedback"] == "同意":
        return dispatch(state)
    return "plan"  # 不同意：带着意见回去重新规划


# ===== 第 3 段：并行检索 =====
def dispatch(state: State) -> list[Send]:
    # TODO 1：每个子问题派一个 research_one（和 Day6 一样）
    sends = []
    for q in state["questions"]:
        sends.append(Send("research_one", {"question": q}))
    return sends


def research_one(state: dict) -> dict:
    question = state["question"]
    pick = get_llm().with_structured_output(Pick).invoke(f"这个问题该用哪个工具查？{question}")
    # TODO 2：pick.tool 是 "kb" 就用 search_kb，否则用 web_search，结果存进 hits
    if pick.tool=="kb":
        hits=search_kb(question)
    else:
        hits=web_search(question)
    print(f"🔎 [{pick.tool}] {question}：{len(hits)} 条")
    return {"hits": hits}


# ===== 第 4 段：反思 =====
def reflect(state: State) -> dict:
    rounds = state["rounds"] + 1
    if rounds >= MAX_ROUNDS:
        return {"rounds": rounds, "enough": True}
    prompt = (
        f"研究主题：{state['topic']}\n已有资料：\n{format_hits(state['hits'])}\n"
        "这些资料够写一份报告吗？不够的话还要搜哪些问题？"
    )
    result = get_llm().with_structured_output(Reflection).invoke(prompt)
    print(f"🤔 反思：够了={result.enough}，补搜={result.questions}")
    return {"rounds": rounds, "enough": result.enough, "questions": result.questions}


def after_reflect(state: State):
    # TODO 3：资料够了（或者没有要补搜的问题）就去 "write_report"，否则 return dispatch(state)
    if state["enough"] or not state["questions"]:
        return "write_report"
    return dispatch(state)


# ===== 第 5 段：写报告 =====
def write_report(state: State) -> dict:
    hits = unique(state["hits"])
    prompt = (
        f"根据下面的资料，写一份 500 字以内的研究报告，主题：{state['topic']}\n"
        "要求：用 Markdown；句子后面用 [编号] 标出用到的资料；只写资料里有的信息。\n\n"
        f"{format_hits(hits)}"
    )
    reply = get_llm().invoke(prompt)
    sources = []
    for i, h in enumerate(hits, 1):
        sources.append(f"[{i}] {h['source']}")
    return {"report": reply.content + "\n\n## 参考来源\n" + "\n".join(sources)}


# ===== 第 6 段：连成图 =====
def build_graph(checkpointer=None):
    graph = StateGraph(State)
    graph.add_node("plan", plan)
    graph.add_node("human_review", human_review)
    graph.add_node("research_one", research_one)
    graph.add_node("reflect", reflect)
    graph.add_node("write_report", write_report)

    graph.add_edge(START, "plan")
    graph.add_edge("plan", "human_review")
    graph.add_conditional_edges("human_review", after_review, ["research_one", "plan"])
    graph.add_edge("research_one", "reflect")  # 所有 research_one 跑完才进 reflect
    graph.add_conditional_edges("reflect", after_reflect, ["research_one", "write_report"])
    graph.add_edge("write_report", END)
    return graph.compile(checkpointer=checkpointer)

"""Day6：拆子问题 -> 并行搜索 -> 汇总成报告。运行：python -m lessons.day6.research_graph"""

import operator
import time
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field

from common.llm import get_llm
from day1_optimized_v5 import search_docs


class Plan(BaseModel):
    questions: list[str] = Field(description="拆出来的子问题，每个都能单独搜索")


class State(TypedDict):
    topic: str
    questions: list[str]
    notes: Annotated[list[str], operator.add]  # 每个子问题的资料，并行写入要累加
    report: str


def plan(state: State) -> dict:
    llm = get_llm().with_structured_output(Plan)
    result = llm.invoke(f"把这个研究主题拆成 3 个子问题：{state['topic']}")
    print("子问题：", result.questions)
    return {"questions": result.questions}


def dispatch(state: State) -> list[Send]:
    # TODO：每个子问题派一个 search_one 去搜，它们会同时运行
    sends = []
    for q in state["questions"]:
        sends.append(Send("search_one", {"question": q}))
    return sends

def search_one(state: dict) -> dict:
    # 注意：这里收到的不是整个 State，而是 Send 里给的那个小字典
    start = time.time()
    text = search_docs(state["question"])
    print(f"搜完：{state['question']}（{time.time() - start:.1f} 秒）")
    return {"notes": [f"【{state['question']}】\n{text}"]}


def write_report(state: State) -> dict:
    notes = "\n\n".join(state["notes"])
    reply = get_llm().invoke(f"根据下面的资料，写一份 300 字以内的研究报告，主题：{state['topic']}\n\n{notes}")
    return {"report": reply.content}


def build_graph():
    graph = StateGraph(State)
    graph.add_node("plan", plan)
    graph.add_node("search_one", search_one)
    graph.add_node("write_report", write_report)

    graph.add_edge(START, "plan")
    graph.add_conditional_edges("plan", dispatch, ["search_one"])
    graph.add_edge("search_one", "write_report")  # 所有 search_one 都跑完，才会进 write_report
    graph.add_edge("write_report", END)
    return graph.compile()


if __name__ == "__main__":
    start = time.time()
    result = build_graph().invoke({"topic": "腾讯有哪些业务"})
    print(f"\n总耗时：{time.time() - start:.1f} 秒")
    print("\n=== 报告 ===")
    print(result["report"])

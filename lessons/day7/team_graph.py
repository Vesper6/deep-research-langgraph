"""Day7：主管 + 3 个员工（研究员、写手、审稿人）。运行：python -m lessons.day7.team_graph"""

import operator
from datetime import date
from typing import Annotated, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from common.llm import get_llm
from lessons.day6.research_graph import build_graph as build_research

MAX_STEPS = 10  # 主管最多安排几次，防止死循环


class State(TypedDict):
    topic: str
    notes: Annotated[list[str], operator.add]
    report: str     # 研究员交回的研究资料
    draft: str      # 写手写的文章
    feedback: str   # 审稿人的意见
    approved: bool  # 审稿人是否通过
    next: str       # 主管决定下一个是谁
    steps: int      # 主管已经安排了几次
    title:str


# ===== 员工 1：研究员 = Day6 的整张图，直接当成一个节点用（子图）=====
researcher = build_research()


# ===== 员工 2：写手 =====
def writer(state: State) -> dict:
    prompt = (
        f"根据研究资料，写一篇 200 字以内的文章，主题：{state['topic']}\n"
        f"研究资料：{state['report']}\n"
        f"审稿意见（如果有，请按意见修改）：{state['feedback']}"
    )
    reply = get_llm().invoke(prompt)
    print("✍️ 写手交稿")
    return {"draft": reply.content,"feedback": "（新稿还没审）"}

def titler(state:State) ->dict:
    prompt = f"给这篇文章起一个 15 字以内的标题，只输出标题：{state['draft']}"
    reply=get_llm().invoke(prompt)
    print("🏷️ 起好标题")
    return  {"title": reply.content}


# ===== 员工 3：审稿人 =====
class Review(BaseModel):
    approved: bool = Field(description="文章是否合格")
    feedback: str = Field(description="不合格时的修改意见，合格时写“无”")


def reviewer(state: State) -> dict:
    llm = get_llm().with_structured_output(Review)
    prompt = (
        f"今天是 {date.today()}。审核下面这篇文章：是否切题、结构是否清楚。"
        f"小问题不用管，没有明显问题就通过。\n{state['draft']}"
    )
    result = llm.invoke(prompt)
    print(f"🔍 审稿：通过={result.approved}，意见={result.feedback}")
    return {"approved": result.approved, "feedback": result.feedback}


# ===== 主管：看当前进度，决定下一个让谁干 =====
class Route(BaseModel):
    next: Literal["researcher", "writer", "reviewer", "FINISH","titler"]
    reason: str = Field(description="为什么这样安排，一句话")


SUPERVISOR_RULES = (
    "你是团队主管，手下有：researcher（研究员）、writer（写手）、reviewer（审稿人）、titler（起标题的）。"
    "流程：先研究，再写作，再审稿；审稿不通过就让写手改；审稿通过后交给 titler 起标题，起好标题就 FINISH。"
    "研究资料只要不是 0 字就算完成，不要重复研究。"
    "标题只要不是 0 字就算起好了，不要重复起标题。"
)


def supervisor(state: State) -> dict:
    if state["steps"] >= MAX_STEPS:
        print("👔 主管：步数用完，结束")
        return {"next": "FINISH"}

    status = (
        f"研究资料：{len(state['report'])} 字\n"
        f"文章：{len(state['draft'])} 字\n"
        f"标题：{len(state['title'])} 字\n"
        f"审稿通过：{state['approved']}，意见：{state['feedback']}"
        
    )
    llm = get_llm().with_structured_output(Route)
    result = llm.invoke(f"{SUPERVISOR_RULES}\n当前进度：\n{status}\n下一步让谁做？")
    print(f"👔 主管：交给 {result.next}（{result.reason}）")
    return {"next": result.next, "steps": state["steps"] + 1}


def route(state: State) -> str:
    return state["next"]


def build_graph():
    graph = StateGraph(State)
    graph.add_node("supervisor", supervisor)
    graph.add_node("researcher", researcher)
    graph.add_node("writer", writer)
    graph.add_node("reviewer", reviewer)
    graph.add_node("titler", titler)

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        route,
        {"researcher": "researcher", "writer": "writer", "reviewer": "reviewer", "titler": "titler", "FINISH": END},
    )
    # 员工干完活，都回去向主管汇报
    graph.add_edge("researcher", "supervisor")
    graph.add_edge("writer", "supervisor")
    graph.add_edge("reviewer", "supervisor")
    graph.add_edge("titler", "supervisor")
    return graph.compile()


if __name__ == "__main__":
    result = build_graph().invoke(
        {
            "topic": "腾讯有哪些业务",
            "notes": [],
            "report": "",
            "draft": "",
            "feedback": "（还没有审稿）",
            "approved": False,
            "next": "",
            "steps": 0,
            "title":""
        }
    )
    print("\n=== 标题 ===", result["title"])
    print("\n=== 最终文章 ===")
    print(result["draft"])

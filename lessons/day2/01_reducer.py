"""Day2-1：Reducer —— 多个节点往同一张纸上写字时，听谁的？

Day1 里每个字段都是“后写覆盖先写”。这在一条直线流程里没问题，
但遇到下面两种情况就不够用了：
  1) 并行：两个节点同一步一起写 context，LangGraph 不知道该留哪个 -> 直接报错
  2) 循环：重试时 find_context 再写一次 context，上一轮的资料就丢了

Reducer 就是给某个字段配一个“合并规则”：新值来了不是覆盖，而是 reducer(旧值, 新值)。
写法：Annotated[类型, 合并函数]

本文件完全离线，直接运行：python lessons/day2/01_reducer.py
"""

import operator
from typing import Annotated, TypedDict

from langgraph.errors import InvalidUpdateError
from langgraph.graph import END, START, StateGraph


# ===== 演示 1：没有 reducer，两个并行节点写同一个字段 -> 报错 =====
class PlainState(TypedDict):
    question: str
    context: str  # 没配 reducer：默认覆盖


def search_web(state) -> dict:
    return {"context": "网页：腾讯有微信、QQ"}


def search_docs(state) -> dict:
    return {"context": "文档：腾讯有王者荣耀"}


def build_fan_out(state_cls):
    """START 同时连到两个搜索节点 = 并行（同一个 super-step 里一起跑）。"""
    graph = StateGraph(state_cls)
    graph.add_node("search_web", search_web)
    graph.add_node("search_docs", search_docs)
    graph.add_edge(START, "search_web")
    graph.add_edge(START, "search_docs")
    graph.add_edge("search_web", END)
    graph.add_edge("search_docs", END)
    return graph.compile()


# ===== 演示 2：配上 operator.add，列表自动拼接 =====
class ListState(TypedDict):
    question: str
    # 合并规则：旧列表 + 新列表。节点要返回列表，而不是字符串
    context: Annotated[list[str], operator.add]


def search_web_list(state) -> dict:
    return {"context": ["网页：腾讯有微信、QQ"]}


def search_docs_list(state) -> dict:
    return {"context": ["文档：腾讯有王者荣耀"]}


# ===== 演示 3：自定义 reducer —— 去重 + 只保留最近 N 条 =====
MAX_KEEP = 3


def merge_recent(old: list[str], new: list[str]) -> list[str]:
    """reducer 就是普通函数：输入(旧值, 新值)，输出合并后的值。"""
    merged = list(old)
    for item in new:
        if item not in merged:  # 去重
            merged.append(item)
    return merged[-MAX_KEEP:]  # 只留最近 MAX_KEEP 条，防止循环里无限变长


class LoopState(TypedDict):
    queries: Annotated[list[str], merge_recent]  # 每轮用过的搜索词
    round: int  # 普通字段：照旧覆盖


def try_search(state: LoopState) -> dict:
    n = state["round"] + 1
    # 第 2 轮故意重复一个词，看去重是否生效
    query = "腾讯 产品" if n == 2 else f"腾讯 产品 第{n}版"
    return {"queries": [query], "round": n}


def route(state: LoopState) -> str:
    return "again" if state["round"] < 5 else "done"


def build_loop():
    graph = StateGraph(LoopState)
    graph.add_node("try_search", try_search)
    graph.add_edge(START, "try_search")
    graph.add_conditional_edges("try_search", route, {"again": "try_search", "done": END})
    return graph.compile()


if __name__ == "__main__":
    print("=== 演示 1：无 reducer + 并行写同一字段 ===")
    try:
        build_fan_out(PlainState).invoke({"question": "腾讯产品？", "context": ""})
    except InvalidUpdateError as e:
        print(f"报错了（这是预期的）：{type(e).__name__}: {str(e).splitlines()[0]}\n")

    print("=== 演示 2：operator.add 把两路结果拼起来 ===")
    graph = StateGraph(ListState)
    graph.add_node("search_web", search_web_list)
    graph.add_node("search_docs", search_docs_list)
    graph.add_edge(START, "search_web")
    graph.add_edge(START, "search_docs")
    graph.add_edge("search_web", END)
    graph.add_edge("search_docs", END)
    # 注意初始值也会参与合并：传 ["用户自带资料"]，结果里就会多一条
    out = graph.compile().invoke({"question": "腾讯产品？", "context": ["用户自带资料"]})
    print(out["context"], "\n")

    print("=== 演示 3：自定义 reducer（去重 + 最多留 3 条）===")
    out = build_loop().invoke({"queries": [], "round": 0})
    print(f"round={out['round']}, queries={out['queries']}")

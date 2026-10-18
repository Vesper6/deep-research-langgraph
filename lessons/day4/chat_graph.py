"""Day4 用的 Agent：和 Day3 一样，只是 build_graph 多了一个 checkpointer 参数。"""

from datetime import date

from langchain_core.messages import SystemMessage
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from common.llm import get_llm
from lessons.day3.my_tools import add, get_weather, multiply, web_search,send_email
from lessons.day8.kb import search_kb

tools = [get_weather, add, multiply, web_search,send_email,search_kb]


def agent(state: MessagesState) -> dict:
    llm = get_llm().bind_tools(tools)
    sys_msg = SystemMessage(f"今天是 {date.today()}")
    reply = llm.invoke([sys_msg] + state["messages"])
    return {"messages": [reply]}


def build_graph(checkpointer=None):
    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=checkpointer)  # 记忆就是在这里装上的

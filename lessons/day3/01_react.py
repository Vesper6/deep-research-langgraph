"""Day3：ReAct Agent。运行：python -m lessons.day3.01_react"""


from datetime import date
from langchain_core.messages import HumanMessage,SystemMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from common.llm import get_llm
from lessons.day3.my_tools import add, get_weather,multiply,web_search
from langgraph.errors import GraphRecursionError

tools = [get_weather, add,multiply,web_search]


def agent(state: MessagesState) -> dict:
    # 模型看完所有消息，决定：直接回答，还是请求调用工具
    llm = get_llm().bind_tools(tools)
    sys_msg=SystemMessage(f"今天是 {date.today()}")
    reply = llm.invoke([sys_msg]+state["messages"])
    return {"messages": [reply]}


def build_graph():
    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode(tools))  # 现成的节点：自动运行工具，返回 ToolMessage

    graph.add_edge(START, "agent")
    # 现成的路由：有 tool_calls 就去 "tools"，没有就去 END
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")  # 工具跑完，回到模型
    return graph.compile()


if __name__ == "__main__":
    app = build_graph()
    question = "腾讯最新发布了什么产品？"
    try:
        result = app.invoke({"messages": [HumanMessage(question)]},{"recursion_limit": 10})
        for m in result["messages"]:
            print(f"[{m.type}]", m.content or m.tool_calls)
    except GraphRecursionError:
        print("步数超限，已停止")

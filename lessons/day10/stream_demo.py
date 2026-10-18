"""Day10：LangGraph 的流式输出。运行：python -m lessons.day10.stream_demo"""

from langgraph.graph import START, MessagesState, StateGraph

from common.llm import get_llm


def chat(state: MessagesState) -> dict:
    return {"messages": [get_llm().invoke(state["messages"])]}


graph = StateGraph(MessagesState)
graph.add_node("chat", chat)
graph.add_edge(START, "chat")
app = graph.compile()

# invoke：等全部写完才返回。stream：模型写一点，就交出一点
for chunk, meta in app.stream({"messages": [("user", "用 100 字介绍一下杭州")]}, stream_mode="messages"):
    print(chunk.content, end="", flush=True)
print()

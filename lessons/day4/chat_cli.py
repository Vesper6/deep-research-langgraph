import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from lessons.day4.chat_graph import build_graph
from langgraph.types import Command


conn = sqlite3.connect("data/memory.db", check_same_thread=False)
app = build_graph(SqliteSaver(conn))
tid=input("会话编号：")
config = {"configurable": {"thread_id": tid}}


while True:
    text=input("你：")
    if text =="q":
        break
    result=app.invoke({"messages": [("user", text)]}, config)
    if "__interrupt__" in result:
        print("⏸", result["__interrupt__"][0].value)
        decision=input("同意/不同意：")
        result = app.invoke(Command(resume=decision), config)
    print("AI：", result["messages"][-1].content)

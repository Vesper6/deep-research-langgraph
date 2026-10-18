"""命令行入口。运行：python -m deep_research.cli"""

import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from common.llm import ROOT
from deep_research.graph import build_graph

conn = sqlite3.connect(ROOT / "data" / "research.db", check_same_thread=False)
app = build_graph(SqliteSaver(conn))

tid = input("会话编号：")
config = {"configurable": {"thread_id": tid}}
topic = input("研究主题：")

result = app.invoke({"topic": topic, "feedback": "", "hits": [], "rounds": 0}, config)

# TODO：只要还在等审核，就一直问（审核不同意会重新规划，然后再停下来等你）
while "__interrupt__" in result:
    print("⏸", result["__interrupt__"][0].value)
    decision = input("同意，或者输入修改意见：")
    result = app.invoke(Command(resume=decision), config)

print("\n" + result["report"])

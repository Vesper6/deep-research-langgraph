"""DeepResearch 网页接口。运行：uvicorn deep_research.api:app --reload，然后打开 http://127.0.0.1:8000"""

import json
import sqlite3
import uuid

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command
from pydantic import BaseModel

from common.llm import ROOT
from deep_research.graph import build_graph

conn = sqlite3.connect(ROOT / "data" / "research.db", check_same_thread=False)
research = build_graph(SqliteSaver(conn))
app = FastAPI(title="DeepResearch")


# ===== 第 1 段：浏览器发过来的数据长什么样 =====
class StartRequest(BaseModel):
    topic: str


class ResumeRequest(BaseModel):
    thread_id: str
    decision: str


# ===== 第 2 段：打开网页 =====
@app.get("/")
def index():
    return FileResponse(ROOT / "deep_research" / "static" / "index.html")


# ===== 第 3 段：输入主题 → 拆好子问题，停下来等审核 =====
@app.post("/start")
def start(req: StartRequest):
    thread_id = str(uuid.uuid4())  # 每次研究一个新编号，自动生成，不用用户填
    # TODO 1：仿照 cli.py 写 config
    config = {"configurable": {"thread_id": thread_id}} 
    result = research.invoke({"topic": req.topic, "feedback": "", "hits": [], "rounds": 0}, config)
    return {"thread_id": thread_id, "questions": result["questions"]}


# ===== 第 4 段：审核完 → 一边跑一边把进度推给浏览器（SSE 流式）=====
def sse(data: dict) -> str:
    """SSE 规定的格式：data: 内容 + 空一行"""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def describe(node: str, update: dict) -> str:
    """把节点的输出翻译成一句给人看的进度"""
    if node == "plan":
        return "📝 按你的意见重新规划了子问题"
    if node == "research_one":
        return f"🔎 搜到 {len(update['hits'])} 条资料"
    if node == "reflect":
        if update.get("enough"):
            return "🤔 反思：资料够了，开始写报告"
        return f"🤔 反思：还要补搜 {update.get('questions')}"
    return ""


@app.post("/resume")
def resume(req: ResumeRequest):
    config = {"configurable": {"thread_id": req.thread_id}}

    def events():
        stream = research.stream(Command(resume=req.decision), config, stream_mode=["updates", "messages"])
        for mode, data in stream:
            if mode == "updates":  # 某个节点跑完了
                for node, update in data.items():
                    text = describe(node, update or {})
                    if text:
                        yield sse({"type": "step", "text": text})
            if mode == "messages":  # 模型写出了一小段字
                chunk, meta = data
                # TODO 2：只推送 write_report 节点写的字（其他节点的模型输出不给用户看）
                if meta["langgraph_node"] == "write_report":
                    yield sse({"type": "token", "text": chunk.content})

        state = research.get_state(config)
        if state.next:  # 还没跑完：不同意时重新规划了，又停在审核
            yield sse({"type": "review", "questions": state.values["questions"]})
        else:
            yield sse({"type": "done", "report": state.values["report"]})

    return StreamingResponse(events(), media_type="text/event-stream")

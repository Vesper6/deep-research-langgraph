"""Day2 作业。运行：python -m lessons.day2.day2_homework"""

import operator
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from common.llm import get_llm
from day1_optimized_v5 import NO_RESULT_TEXT, search_docs

MAX_RETRIES = 2
FORCE_RETRY = False  # 测试用：强制重试。做完 TODO 4 改成 False

SYSTEM_RULES = (
    "你只能根据用户提供的资料回答问题。"
    "如果资料不足，只输出：资料不足，无法根据现有资料回答。"
    "不要编造资料里没有的内容。"
)


class State(TypedDict):
    question: str
    context: Annotated[list[str], operator.add]  # ✅ TODO 1 已完成
    queries: Annotated[list[str], operator.add]  # ✅ TODO 1 已完成
    prompt: str
    answer: str
    retry_count: int


def rewrite_query(question: str, retry_count: int) -> str:
    # TODO 2：每一轮返回不一样的搜索词
    suffixes=["","官网","业务介绍"]
    return question+suffixes[retry_count % len(suffixes)]


def find_context(state: State) -> dict:
    query = rewrite_query(state["question"], state["retry_count"])
    text = search_docs(query)
    print(f"第 {state['retry_count']} 轮，搜索词：{query}")
    # TODO 1b：把 text 和 query 变成列表
    return {"context": [text], "queries": [query]}


def make_prompt(state: State) -> dict:
    # TODO 1c：把 context 列表拼成一段文字
    context_text="\n".join(state['context'])
    prompt = f"资料：{context_text}\n问题：{state['question']}"
    return {"prompt": prompt}


def call_llm(state: State) -> dict:
    # TODO 3：用 get_llm() 调模型
    llm=get_llm()
    try:
        msgs=[SystemMessage(SYSTEM_RULES),HumanMessage(state["prompt"])]
        reply=llm.invoke(msgs)
        return {"answer": reply.content}
    except Exception as e:
        return{"answer": f"调用模型失败：{e}"}

def bump_retry(state: State) -> dict:
    return {"retry_count": state["retry_count"] + 1}


def is_insufficient(answer: str, context) -> bool:
    if FORCE_RETRY:
        return True
    # TODO 4：context 是列表了，这行还对吗？
    if context.count(NO_RESULT_TEXT) == len(context):
        return True
    keywords = ["资料不足", "未检索到", "无法提供", "无法准确", "缺乏"]
    return any(word in answer for word in keywords)


def route_after_answer(state: State) -> str:
    if is_insufficient(state["answer"], state["context"]) and state["retry_count"] < MAX_RETRIES:
        return "retry"
    return "done"


def build_graph():
    graph = StateGraph(State)
    graph.add_node("find_context", find_context)
    graph.add_node("make_prompt", make_prompt)
    graph.add_node("call_llm", call_llm)
    graph.add_node("bump_retry", bump_retry)

    graph.add_edge(START, "find_context")
    graph.add_edge("find_context", "make_prompt")
    graph.add_edge("make_prompt", "call_llm")
    graph.add_conditional_edges("call_llm", route_after_answer, {"retry": "bump_retry", "done": END})
    graph.add_edge("bump_retry", "find_context")
    return graph.compile()


if __name__ == "__main__":
    app = build_graph()
    result = app.invoke(
        {
            "question": "腾讯的主要产品有哪些？",
            "context": [],  # TODO 1a：初始值改成什么？
            "queries": [],  # TODO 1a：初始值改成什么？
            "prompt": "",
            "answer": "",
            "retry_count": 0,
        }
    )

    print("\n=== 结果 ===")
    print("queries =", result["queries"])
    print("answer  =", result["answer"][:200])


# 思考题：
# 1. 并行节点谁先写，顺序固定吗？
#    答：不固定
# 2. 为什么 messages 用 add_messages，不用 operator.add？
#    答：+ 只能追加；add_messages还能按id替换和删除消息，比如对话太长时删掉旧消息。

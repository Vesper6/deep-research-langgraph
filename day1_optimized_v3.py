"""第一天优化版 v3：LangGraph 骨架 + 本地 Ollama + 联网检索 + 资料不足自动重试一次。

流程：make_plan -> find_context -> make_prompt -> call_llm
      call_llm --(资料不足且未超重试上限)--> bump_retry -> find_context（再试一次）
      call_llm --(其他情况)--> END
- make_plan: 先定一个固定计划
- find_context: 通过工具箱调用 search_docs 联网找资料
- make_prompt: 把问题 + 资料拼成给模型的提示词
- call_llm: 调用本地 Ollama，把回答放进 answer
- bump_retry: 每次决定重试时，先把 retry_count 加 1
  为什么靠它防死循环：retry_count 每重试一次就 +1，
  达到 MAX_RETRIES 后 route_after_answer 不再返回 retry，一定走 done，
  所以最多重试 1 次就会结束，不会无限转圈。
"""

from typing import TypedDict

import requests
from duckduckgo_search import DDGS
from langgraph.graph import END, START, StateGraph

# ===== 1) 配置：集中放最上面，改模型/数量只改这里 =====
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "openbmb/minicpm-v4.6:latest"
MAX_RESULTS = 3  # 每次检索最多取几条结果
MAX_RETRIES = 1  # 资料不足时最多自动重试几次
NO_RESULT_TEXT = "（未检索到相关资料）"


# ===== 2) 状态：图里流转的“文件袋”，每张纸都是文字（retry_count 是数字）=====
class State(TypedDict):
    question: str  # 用户的原始问题
    plan: str      # 做题计划
    context: str   # 找到的资料
    prompt: str    # 拼好后给模型看的话
    answer: str    # 模型返回的最终回答
    retry_count: int  # 已经重试了几次，从 0 开始，每重试一次 +1


# ===== 3) 工具：先定义工具，再登记进工具箱 =====
# 工具和节点不同：工具谁调用谁传参数，直接返回资料本身。
def search_docs(query: str) -> str:
    """按传入的问题联网搜索，返回拼好的一段资料文字。"""
    # 用传进来的问题搜；问题为空时，用一个通用问题兜底
    search_query = query.strip() if query and query.strip() else "腾讯 主要产品"
    try:
        results = DDGS().text(search_query, max_results=MAX_RESULTS) or []
    except Exception:
        # 断网、搜索服务异常时不让程序崩掉，返回一句兜底文字
        return NO_RESULT_TEXT

    texts: list[str] = []
    for r in results:
        body = (r.get("body") or "").strip()
        if body:
            texts.append(body)

    if not texts:
        return NO_RESULT_TEXT
    return "\n".join(texts)


# 工具箱：按名字喊工具。放在 find_context 上面，读代码时先见工具再见调用
TOOL_REGISTRY = {"search_docs": search_docs}


# ===== 4) 节点：每个节点只做一件事，交回要更新的纸 =====
def make_plan(state: State) -> dict[str, str]:
    return {"plan": "先找资料，再回答"}


def find_context(state: State) -> dict[str, str]:
    # 通过工具箱按名字拿到工具，再把问题传给它干活
    return {"context": TOOL_REGISTRY["search_docs"](state["question"])}


def make_prompt(state: State) -> dict[str, str]:
    # 在查完资料之后拼提示词，这样 prompt 里能带上 context
    prompt = (
        "请结合下面资料回答问题。\n"
        "如果资料不足，请说明资料不足，不要编造。\n"
        f"资料：{state['context']}\n"
        f"问题：{state['question']}"
    )
    return {"prompt": prompt}


def call_llm(state: State) -> dict[str, str]:
    try:
        r = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": state["prompt"],
                "stream": False,
            },
            timeout=60,
        )
        r.raise_for_status()
        data = r.json()
        answer = (data.get("response") or "").strip()
        # 有些模型会先输出思考过程，用 </think> 分隔，只留最后的正式回答
        if "</think>" in answer:
            answer = answer.split("</think>")[-1].strip()
        return {"answer": answer}
    except requests.RequestException as e:
        # Ollama 没启动 / 超时 / 连不上时，不让整个程序崩掉
        return {"answer": f"[调用本地模型失败，请确认 Ollama 已启动] {e}"}


def bump_retry(state: State) -> dict[str, int]:
    # 重试前先记账：次数 +1。这样下一次判断时就能靠次数停下来，
    # 不必依赖模型下一句回答到底用了哪个词。
    return {"retry_count": state["retry_count"] + 1}


# ===== 5) 路由判断：回答完后决定是重试还是结束 =====
def is_insufficient(answer: str, context: str) -> bool:
    """资料是否不足：没搜到资料，或者回答里出现了“说不清”的信号词。"""
    if context == NO_RESULT_TEXT:
        return True
    keywords = ["资料不足", "未检索到", "无法提供", "无法准确", "缺乏"]
    return any(word in answer for word in keywords)


def route_after_answer(state: State) -> str:
    # 两个条件都成立才重试：资料确实不足，并且还没用完重试次数。
    # 每走一次 retry，bump_retry 都会把 retry_count 加 1；
    # 加到 MAX_RETRIES 后这里就会改走 done，循环自然结束。
    if is_insufficient(state["answer"], state["context"]) and state["retry_count"] < MAX_RETRIES:
        return "retry"
    return "done"


# ===== 6) 建图：登记节点 + 连线 + 编译 =====
def build_graph():
    graph = StateGraph(State)

    graph.add_node("make_plan", make_plan)
    graph.add_node("find_context", find_context)
    graph.add_node("make_prompt", make_prompt)
    graph.add_node("call_llm", call_llm)
    graph.add_node("bump_retry", bump_retry)

    graph.add_edge(START, "make_plan")
    graph.add_edge("make_plan", "find_context")
    graph.add_edge("find_context", "make_prompt")
    graph.add_edge("make_prompt", "call_llm")
    # 注意：call_llm 后面只走下面这条条件边，不再另加一条普通边到 END，
    # 否则同一起点会有两条出边，到底走哪条就不清楚了。
    graph.add_conditional_edges(
        "call_llm",
        route_after_answer,
        {"retry": "bump_retry", "done": END},
    )
    # 重试时先经 bump_retry 把次数 +1，再回 find_context 重新找资料
    graph.add_edge("bump_retry", "find_context")

    return graph.compile()


# ===== 7) 直接运行本文件时的小演示 =====
if __name__ == "__main__":
    app = build_graph()

    result = app.invoke(
        {
            "question": "腾讯的主要产品有哪些？",
            "plan": "",
            "context": "",
            "prompt": "",
            "answer": "",
            "retry_count": 0,
        }
    )

    print("=== 最终回答 ===")
    print(result["answer"])
    print("\n=== 完整状态 ===")
    print(result)

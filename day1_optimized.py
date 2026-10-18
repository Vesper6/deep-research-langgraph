"""第一天优化版：LangGraph 骨架 + 本地 Ollama 真实调用。

流程：make_plan -> find_context -> make_prompt -> call_llm -> END
- make_plan: 先定一个固定计划
- find_context: 先给一份假资料（第三天会换成真检索工具）
- make_prompt: 把问题 + 资料拼成给模型的提示词
- call_llm: 调用本地 Ollama，把回答放进 answer
"""

from typing import TypedDict

import requests
from langgraph.graph import END, START, StateGraph
from duckduckgo_search import DDGS


# ===== 1) 配置：集中放最上面，改模型只改这里 =====
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "openbmb/minicpm-v4.6:latest"


# ===== 2) 状态：图里流转的“文件袋”，每张纸都是文字 =====
class State(TypedDict):
    question: str  # 用户的原始问题
    plan: str      # 做题计划
    context: str   # 找到的资料
    prompt: str    # 拼好后给模型看的话
    answer: str    # 模型返回的最终回答


# ===== 3) 节点：每个节点只做一件事，交回要更新的纸 =====
def make_plan(state: State) -> dict[str, str]:
    return {"plan": "先找资料，再回答"}


def find_context(state: State) -> dict[str, str]:
    return {"context": TOOL_REGISTRY["search_docs"]((state["question"]))}


def make_prompt(state: State) -> dict[str, str]:
    # 在查完资料之后拼提示词，这样 prompt 里能带上 context
    prompt = (
        "请结合下面资料回答问题。\n"
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
        # 用 .get 兜底，避免模型没返回 response 时直接 KeyError
        return {"answer": (data.get("response") or "").strip()}
    except requests.RequestException as e:
        # Ollama 没启动 / 超时 / 连不上时，不让整个程序崩掉
        return {"answer": f"[调用本地模型失败，请确认 Ollama 已启动] {e}"}


def search_docs(query) -> str:
    results=DDGS().text("腾讯 主要产品", max_results=3)
    #print(results)
    texts=[]
    for r in results:
        texts.append(r["body"])
    return "\n".join(texts)
    
TOOL_REGISTRY = {"search_docs": search_docs}

# ===== 4) 建图：登记节点 + 连线 + 编译 =====
def build_graph():
    graph = StateGraph(State)

    graph.add_node("make_plan", make_plan)
    graph.add_node("find_context", find_context)
    graph.add_node("make_prompt", make_prompt)
    graph.add_node("call_llm", call_llm)

    graph.add_edge(START, "make_plan")
    graph.add_edge("make_plan", "find_context")
    graph.add_edge("find_context", "make_prompt")
    graph.add_edge("make_prompt", "call_llm")
    graph.add_edge("call_llm", END)

    return graph.compile()


# ===== 5) 直接运行本文件时的小演示 =====
if __name__ == "__main__":
    app = build_graph()

    result = app.invoke(
        {
            "question": "腾讯的主要产品有哪些？",
            "plan": "",
            "context": "",
            "prompt": "",
            "answer": "",
        }
    )

    print("=== 最终回答 ===")
    print(result["answer"])
    print("\n=== 完整状态 ===")
    print(result)

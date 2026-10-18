"""第一天优化版 v5：LangGraph 骨架 + 本地 Ollama + 联网检索 + 资料不足自动重试一次。

在 v4 基础上整理：
- 配置不再写死在代码里，统一从 config/local.yaml 读取（见 load_config）。
- 运行结束在 __main__ 底部写 logs/runs.log，带 status 方便以后算成功率。

流程：make_plan -> find_context -> make_prompt -> call_llm
      call_llm --(资料不足且未超重试上限)--> bump_retry -> find_context（再试一次）
      call_llm --(其他情况)--> END
- make_plan: 先定一个固定计划
- find_context: 通过工具箱调用 search_docs 联网找资料
- make_prompt: 把问题 + 资料拼成给模型的提示词
- call_llm: 调用本地 Ollama，把回答放进 answer
- 搜索改用新包 ddgs（旧 duckduckgo_search 已改名且实测不稳定，详见 search_docs 注释）
- bump_retry: 每次决定重试时，先把 retry_count 加 1
  为什么靠它防死循环：retry_count 每重试一次就 +1，
  达到 search.max_retries 后 route_after_answer 不再返回 retry，一定走 done，
  所以最多重试配置的次数就会结束，不会无限转圈。
"""

import os
from datetime import datetime
from typing import TypedDict

import requests
import yaml

try:
    from ddgs import DDGS  # 新包：pip install ddgs（旧 duckduckgo_search 已改名）
except ImportError:  # 兜底：还没装新包时仍能用旧包跑，但搜索可能不稳定
    from duckduckgo_search import DDGS
from langgraph.graph import END, START, StateGraph

# ===== 1) 配置：从 config/local.yaml 读取，不在代码里写死 =====
CONFIG_PATH = "config/local.yaml"  # 相对运行目录：请在项目根目录运行本文件

# 每段必须有的 key；缺了就在 load_config 里友好报错，指出缺哪一段/哪个 key。
_REQUIRED_KEYS = {
    "model": ("name", "api_url", "timeout"),
    "search": ("max_results", "max_retries"),
}


def load_config(path: str = CONFIG_PATH) -> dict:
    """读取并校验 YAML 配置，返回 config 字典；缺段/缺 key 时说明缺什么。"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError as e:
        raise SystemExit(
            f"[config] 找不到配置文件 {path!r}，请确认在项目根目录运行，"
            f"且 config/local.yaml 已存在。"
        ) from e
    except yaml.YAMLError as e:
        raise SystemExit(f"[config] 配置文件 {path!r} 不是合法 YAML：{e}") from e

    if not isinstance(config, dict):
        raise SystemExit(f"[config] {path!r} 顶层应是字典，包含 model / search 两段。")

    for section, keys in _REQUIRED_KEYS.items():
        if section not in config or not isinstance(config[section], dict):
            raise SystemExit(
                f"[config] 缺少配置段 {section!r}，请在 {path!r} 里补上 {section}: 段。"
            )
        missing = [key for key in keys if key not in config[section]]
        if missing:
            raise SystemExit(
                f"[config] 配置段 {section!r} 缺少 key: {', '.join(missing)}，"
                f"请在 {path!r} 的 {section}: 段里补齐。"
            )
    return config


CONFIG = load_config()

NO_RESULT_TEXT = "（未检索到相关资料）"


# ===== 2) 状态：图里流转的“文件袋”，每张纸都是文字（retry_count 是数字）=====
class State(TypedDict):
    question: str  # 用户的原始问题
    plan: str  # 做题计划
    context: str  # 找到的资料
    prompt: str  # 拼好后给模型看的话
    answer: str  # 模型返回的最终回答
    retry_count: int  # 已经重试了几次，从 0 开始，每重试一次 +1


# ===== 3) 工具：先定义工具，再登记进工具箱 =====
# 工具和节点不同：工具谁调用谁传参数，直接返回资料本身。
# 调试开关：改成 True 时打印每次搜到的原始结果条数，分清“没搜到”还是“被丢掉”。
DEBUG_SEARCH = False


def _search_once(search_query: str) -> list:
    """搜一次，返回原始结果列表；失败时把异常打出来（不再静默吞掉），再交回空列表。"""
    try:
        # 新包推荐的上下文写法：用完自动关闭连接，连续重试时更稳定
        with DDGS() as ddgs:
            results = ddgs.text(
                search_query, max_results=CONFIG["search"]["max_results"]
            ) or []
    except Exception as e:
        print(f"[search_docs] 搜索失败 query={search_query!r}: {type(e).__name__}: {e}")
        return []
    if DEBUG_SEARCH:
        print(f"[search_docs] query={search_query!r} 原始结果 {len(results)} 条: {results}")
    return results


def search_docs(query: str) -> str:
    """按传入的问题联网搜索，返回拼好的一段资料文字。"""
    # 用传进来的问题搜；问题为空时，用一个通用问题兜底
    search_query = query.strip() if query and query.strip() else "腾讯 主要产品"

    # 候选问法：先用原问题；搜不到时再换一个去掉“有哪些”的短问法。
    # 旧包对同一问题会时有时无地返回 []，新包更稳，这里再加一层换问法重试。
    candidates = [search_query]
    short_query = search_query.rstrip("？?。").replace("有哪些", "").strip()
    if short_query and short_query not in candidates:
        candidates.append(short_query)

    results: list = []
    for candidate in candidates:
        results = _search_once(candidate)
        if results:
            break
        # 同一问法立刻再试一次，躲开搜索服务偶发的空返回
        results = _search_once(candidate)
        if results:
            break

    texts: list[str] = []
    for r in results:
        # 正文优先；正文为空时用标题兜底，避免搜到了却因为 body 为空被丢掉
        text = (r.get("body") or r.get("title") or "").strip()
        if text:
            texts.append(text)

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
        "请只结合下面资料回答问题。\n"
        "硬规则：如果资料为空、是（未检索到相关资料），或不足以回答问题，"
        "你只能输出这一句：资料不足，无法根据现有资料回答。"
        "输出这句后立即停止，不要补充、猜测或列举任何产品名、公司名或例子。\n"
        "如果资料充足，只使用资料中明确出现的信息回答，不要补充资料外的内容。\n"
        f"资料：{state['context']}\n"
        f"问题：{state['question']}"
    )
    return {"prompt": prompt}


def call_llm(state: State) -> dict[str, str]:
    try:
        r = requests.post(
            CONFIG["model"]["api_url"],
            json={
                "model": CONFIG["model"]["name"],
                "prompt": state["prompt"],
                "stream": False,
            },
            timeout=CONFIG["model"]["timeout"],
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
    # 重试前先记账：次数 +1。下一次判断就能靠次数停下来，不依赖模型措辞。
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
    # 加到 max_retries 后这里改走 done，循环自然结束（防死循环）。
    if (
        is_insufficient(state["answer"], state["context"])
        and state["retry_count"] < CONFIG["search"]["max_retries"]
    ):
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
    # call_llm 后只走这条条件边，不再另加普通边到 END，否则出边不明确。
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

    # 日志只能写在这里：此时 result 已产生，import 区写日志会因 result 不存在而报错。
    # status 复用 is_insufficient 判断，方便以后按 success/failed 数行数算成功率。
    status = "failed" if is_insufficient(result["answer"], result["context"]) else "success"
    os.makedirs("logs", exist_ok=True)
    with open("logs/runs.log", "a", encoding="utf-8") as log:
        log.write(
            f"{datetime.now()} | {result['question']} | "
            f"retry={result['retry_count']} | status={status}\n"
        )

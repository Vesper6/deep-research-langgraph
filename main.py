"""LangGraph 最小入门例子：完全离线，不需要任何 LLM API Key。

流程图：
    START -> count_words -> (条件边)
                               ├─ 词数 > 5 -> summarize -> END
                               └─ 词数 <= 5 -> shorten  -> END
"""

from typing import TypedDict

from langgraph.graph import StateGraph, START, END


# 1) State（状态）：整张图共享的数据，用 TypedDict 定义。
#    每个节点都会收到 State，并返回要更新的字段（不用返回全部）。
class State(TypedDict):
    text: str        # 输入原文
    word_count: int  # count_words 节点算出的词数
    result: str      # 最终结果
    word_info: str   #共 word_count 个词
    attempts: int    


# 2) Nodes（节点）：就是普通 Python 函数，输入是 State，输出是 dict。
def count_words(state: State) -> dict:
    """统计词数：英文按空格分词即可，确定性、无外部调用。"""
    count = len(state["text"].split())
    print(f"[count_words] text={state['text']!r} -> word_count={count}")
    return {"word_count": count}


def summarize(state: State) -> dict:
    """长文本分支：离线“假摘要”——取第一句 + 截断，不调用大模型。"""
    text = state["text"]
    first_sentence = text.split(".")[0].strip()
    if first_sentence:
        summary = first_sentence + "."
    else:
        summary = text[:40]
    print(f"[summarize] -> {summary!r}")
    return {"result": f"摘要: {summary}"}


def shorten(state: State) -> dict:
    """短文本分支：不用摘要，直接压缩空白并标注。"""
    shortened = " ".join(state["text"].split())
    print(f"[shorten] -> {shortened!r}")
    return {"result": f"短文本: {shortened}"}


# 3) 条件边逻辑：根据 State 决定下一步去哪个节点。
def route_by_length(state: State) -> str:
    # 返回的字符串必须和下面 add_conditional_edges 里的映射 key 一致
    return "summarize" if state["word_count"] > 5 else "shorten"



def add_prefix(state:State) -> dict:
    return {"result":"[done] "+state['result']}
def describe_words(state:State) -> dict:
    return {"word_info":f"共{state['word_count']}个词"}
def make_upper(state:State) -> dict:
    return {"result":state['result'].upper()}
def add_exclaim(state:State) -> dict:
    return {"result":state['result']+"!","attempts":state['attempts']+1}

def route_by_attempts(state: State) -> str:
    return  "add_exclaim" if state["attempts"] <3 else "add_prefix"

# 4) 建图 -> 编译 -> 调用
def build_graph():
    graph = StateGraph(State)

    # 添加节点：名字 + 函数
    graph.add_node("count_words", count_words)
    graph.add_node("describe_words", describe_words)
    graph.add_node("summarize", summarize)
    graph.add_node("shorten", shorten)
    graph.add_node("add_prefix",add_prefix)
    graph.add_node("make_upper",make_upper)
    graph.add_node("add_exclaim",add_exclaim)
    

    # 普通边：固定走向
    graph.add_edge(START, "count_words")
    graph.add_edge("summarize", "add_prefix")
    graph.add_edge("shorten", "make_upper")
    graph.add_edge("make_upper", "add_exclaim")
    graph.add_edge("add_prefix", "describe_words")
    graph.add_edge("describe_words", END )
    # 条件边：从 count_words 出发，由 route_by_length 决定分支
    graph.add_conditional_edges(
        "count_words",
        route_by_length,
        {"summarize": "summarize", "shorten": "shorten"},
    )
    graph.add_conditional_edges(
        "add_exclaim",
        route_by_attempts,
        {"add_exclaim": "add_exclaim", "add_prefix": "add_prefix"}
    )

    # compile() 把图定义变成可运行的应用
    return graph.compile()


if __name__ == "__main__":
    app = build_graph()

    print("=== 测试 1：长文本（应走 summarize）===")
    out1 = app.invoke({
        "text": "LangGraph is a framework for building agents. It uses graphs.",
        "word_count": 0,
        "result": "",
        "attempts": 0
    })
    print(f"最终结果: {out1}\n")

    print("=== 测试 2：短文本（应走 shorten）===")
    out2 = app.invoke({
        "text": "Hello   LangGraph world",
        "word_count": 0,
        "result": "",
        "attempts": 0
    })
    print(f"最终结果: {out2}")

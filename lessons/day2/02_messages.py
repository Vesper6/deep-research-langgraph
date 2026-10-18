"""Day2-2：消息（messages）+ 聊天模型 —— Agent 的标准“文件袋”

Day1 我们手拼 prompt 字符串、用 requests 调 Ollama。真实 Agent 几乎都改用“消息列表”：
  SystemMessage  系统设定（你是谁、守什么规矩）
  HumanMessage   用户说的话
  AIMessage      模型说的话（以后还会带 tool_calls，Day3 讲）
  ToolMessage    工具执行结果（Day3 讲）

LangGraph 给消息列表准备了专用 reducer：add_messages
  - 新消息默认追加到末尾（像 operator.add）
  - 但如果新消息的 id 和已有消息相同 -> 原地替换（operator.add 做不到）
  - 返回 RemoveMessage(id=...) -> 删除那条消息（用来裁剪历史）
MessagesState 就是内置好的 {"messages": Annotated[list, add_messages]}。

运行（在项目根目录，用 -m 方式，这样才能 import common）：
    python -m lessons.day2.02_messages
第 1 部分离线；第 2 部分按 config/local.yaml 的 llm.provider 调 DeepSeek 或本地 Ollama。
"""

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.message import add_messages

from common.llm import get_llm


# ===== 第 1 部分：离线看清 add_messages 的三种行为 =====
def demo_add_messages() -> None:
    history = [
        HumanMessage("你好", id="1"),
        AIMessage("你好！有什么可以帮你？", id="2"),
    ]

    # 行为 1：追加（新 id）
    history = add_messages(history, [HumanMessage("腾讯有哪些产品？", id="3")])
    print("追加后：", [m.content for m in history])

    # 行为 2：替换（同 id）。比如想修正模型上一句回答
    history = add_messages(history, [AIMessage("（已修正的回答）", id="2")])
    print("替换后：", [m.content for m in history])

    # 行为 3：删除。对话太长时用它裁剪旧消息，控制喂给模型的长度
    history = add_messages(history, [RemoveMessage(id="1")])
    print("删除后：", [m.content for m in history])


# ===== 第 2 部分：一个最小的聊天图 =====
# 用 LangChain 的聊天模型代替手写 requests。具体是 DeepSeek 还是 Ollama，
# 由 config/local.yaml 的 llm.provider 决定（见 common/llm.py），节点代码不用动。
# 本机代理会拦 Ollama 的 localhost 请求（502），这个坑也在 common/llm.py 里统一处理了。

SYSTEM_PROMPT = SystemMessage("你是一个简洁的中文助手，每次回答不超过 50 字。")


def chatbot(state: MessagesState) -> dict:
    # 在节点里才拿模型（get_llm 有缓存，只建一次）：没配 Key 时第 1 部分照样能离线跑
    llm = get_llm()
    # 系统设定不存进 state，每次调用时临时放在最前面，避免历史里重复堆积
    reply = llm.invoke([SYSTEM_PROMPT] + state["messages"])
    # 只返回“新增的那条”，add_messages 会负责追加
    return {"messages": [reply]}


def build_graph():
    graph = StateGraph(MessagesState)
    graph.add_node("chatbot", chatbot)
    graph.add_edge(START, "chatbot")
    graph.add_edge("chatbot", END)
    return graph.compile()


def demo_chat() -> None:
    app = build_graph()

    # 第一轮
    state = app.invoke({"messages": [HumanMessage("我叫小林，我在学 LangGraph。")]})
    print("AI：", state["messages"][-1].content)

    # 第二轮：把上一轮的完整 state 带回去，再追加新问题 -> 模型就“记得”上文
    # （Day4 会用 checkpointer 自动保存，不用自己手动传来传去）
    state["messages"].append(HumanMessage("我叫什么？我在学什么？"))
    state = app.invoke(state)
    print("AI：", state["messages"][-1].content)

    print("\n--- 完整消息历史 ---")
    for m in state["messages"]:
        print(f"[{m.type}] {m.content}")


if __name__ == "__main__":
    print("=== 第 1 部分：add_messages 行为（离线）===")
    demo_add_messages()

    print("\n=== 第 2 部分：多轮聊天（调用大模型）===")
    try:
        demo_chat()
    except (Exception, SystemExit) as e:  # SystemExit：get_llm 发现没配 Key
        print(f"[调用模型失败：检查 .env 里的 Key，或 Ollama 是否已启动] {type(e).__name__}: {e}")

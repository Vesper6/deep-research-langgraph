# Day2 练习：用今天学的东西改造 v5

在 `lessons/day2/` 下新建 `day2_homework.py`，可以从 `day1_optimized_v5.py` 复制一份开始改。

## 任务 1：让重试“真的换个方式搜”（练 Reducer）

现状问题：v5 重试时 `find_context` 拿同一个问题再搜一次，结果大概率不变，重试等于白做。

要求：
1. State 里加一个字段 `queries: Annotated[list[str], operator.add]`，记录每一轮实际用过的搜索词
2. `find_context` 根据 `retry_count` 选不同的搜索词：第 0 轮用原问题，第 1 轮用改写后的问题
   （改写方法自己定，最简单的是加关键词，例如 `问题 + " 官网 介绍"`）
3. `find_context` 返回 `{"context": ..., "queries": [这次用的词]}`
4. 把 `context` 也改成累加：`Annotated[list[str], operator.add]`，这样重试后两轮资料都保留；
   `make_prompt` 里用 `"\n".join(state["context"])` 拼起来

验收：故意把 `max_retries` 设成 2，并让 `is_insufficient` 永远返回 True，
跑完后 `result["queries"]` 应该有 3 个**不同**的搜索词。

## 任务 2：把 call_llm 换成 get_llm() + 消息（练 messages）

1. 用 `from common.llm import get_llm` 拿模型，代替 `requests.post`
   （运行方式：`python -m lessons.day2.day2_homework`）
2. 把 prompt 拆成两条消息：硬规则放 `SystemMessage`，资料 + 问题放 `HumanMessage`
3. `answer` 字段保持字符串不变：`{"answer": reply.content}`

## 思考题（写在代码注释里即可）

1. 演示 2 中，最终结果里“文档”排在“网页”前面。并行节点的写入顺序有保证吗？
   如果要求顺序固定，你会怎么做？
2. 为什么 `messages` 用 `add_messages`，而不直接用 `operator.add`？举一个只有 add_messages 能做到的场景。
3. `02_messages.py` 第二轮里，模型回答“我叫小林”。这个错误出在图的设计、提示词，还是模型本身？怎么验证？

做完把文件路径告诉我，我来 review，然后进入 Day3。

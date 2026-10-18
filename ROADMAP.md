# LangGraph 学习路线 → 简历项目

## 最终项目：DeepResearch —— 基于 LangGraph 的多智能体研究助手

输入一个研究问题，系统自动：拆解子问题 → 并行联网/本地知识库检索 → 反思补搜 →
人工审核大纲 → 撰写带引用的研究报告；支持多轮追问、断点续跑、流式输出。

简历上能写的技术点（每一条都对应下面某一天）：

- LangGraph 状态图编排：条件路由、循环重试、Reducer 状态合并
- ReAct 工具调用 Agent：bind_tools + ToolNode，可插拔工具注册
- Map-Reduce 并行检索：Send API 动态扇出子任务
- 多智能体协作：Supervisor + 子图（规划 / 研究 / 写作 / 审稿）
- 持久化记忆：SQLite Checkpointer，thread 级会话、断点续跑、时间回溯
- Human-in-the-loop：interrupt 人工审核研究大纲
- Agentic RAG：本地文档向量检索 + 相关性评分 + 查询改写
- 工程化：FastAPI + SSE 流式接口、Web 界面、配置管理、单元测试、评测集、Docker

## 进度

| 天 | 主题 | 产出 | 状态 |
|---|---|---|---|
| Day1 | StateGraph 基础：节点、边、条件边、循环 | `day1_optimized_v5.py` | ✅ |
| Day2 | Reducer、消息、聊天模型 | `lessons/day2/` | ✅ |
| Day3 | 工具调用与 ReAct Agent | `lessons/day3/` | ✅ |
| Day4 | Checkpointer 记忆、多轮会话、时间回溯 | `lessons/day4/` | ✅ |
| Day5 | Human-in-the-loop：interrupt / Command | `lessons/day5/` | ✅ |
| Day6 | 结构化输出 + Send 并行 Map-Reduce | `lessons/day6/` | ✅ |
| Day7 | 子图与多智能体（Supervisor） | `lessons/day7/` | ✅ |
| Day8 | Agentic RAG：本地知识库 | `lessons/day8/` | ✅ |
| Day9 | 项目整合：`deep_research/` 包结构 | `deep_research/` | ✅ |
| Day10 | 服务化：FastAPI + 流式 + 前端 | `deep_research/api` | ✅ |
| Day11 | 测试、评测、可观测性（LangSmith） | `tests/`、`eval/` | ✅ |
| Day12 | Docker、README、简历描述 | 完整仓库 | 下一步 |

## 环境备忘

- 用 Windows 的 Python 运行：`.venv\Scripts\python.exe`（这个 venv 是 Windows 下建的）
- 课程在项目根目录用 `-m` 方式运行：`python -m lessons.day2.02_messages`（这样才能 `import common`）
- 大模型：默认 DeepSeek（`deepseek-chat`），统一从 `common/llm.py` 的 `get_llm()` 拿；
  切换到本地 Ollama 只改 `config/local.yaml` 的 `llm.provider`
- API Key 放 `.env`（照 `.env.example` 写），已加进 `.gitignore`
- 系统代理 127.0.0.1:7897 会拦截 httpx 访问 localhost → `common/llm.py` 里已设 `NO_PROXY`

"""统一的大模型入口：所有课程和最终项目都通过 get_llm() 拿模型，不在节点里直接 new。

为什么要单独抽一个文件：
- 换模型只改 config/local.yaml 的 llm.provider，节点代码一行不动
- API Key 统一从 .env 读，不会散落在各个文件里、更不会被提交到 Git
- 以后写测试时，可以在这里换成假模型，不用真花钱调 API

用法（在项目根目录用 -m 方式运行，这样 import common 才找得到）：
    from common.llm import get_llm
    llm = get_llm()
"""

import os
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

# 项目根目录 = 本文件的上一级的上一级；用绝对路径，不再依赖“必须在根目录运行”
ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "local.yaml"

# 本机系统代理会拦截 httpx 访问 localhost（Ollama 返回 502），本机地址一律不走代理。
# DeepSeek 是外网地址，照常走代理，不受影响。
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")

# 读取 .env 里的 DEEPSEEK_API_KEY 等变量；已存在的系统环境变量优先，不会被覆盖
load_dotenv(ROOT / ".env")


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_llm_config() -> dict:
    config = load_config()
    if "llm" not in config:
        raise SystemExit(f"[llm] {CONFIG_PATH} 里缺少 llm: 配置段。")
    return config["llm"]


@lru_cache  # 同样的参数只建一次模型对象，节点里反复调用 get_llm() 也不浪费
def get_llm(**overrides) -> BaseChatModel:
    """按配置返回一个聊天模型。overrides 可临时覆盖参数，例如 get_llm(temperature=0.7)。"""
    cfg = load_llm_config()
    provider = cfg.get("provider", "deepseek")
    temperature = overrides.pop("temperature", cfg.get("temperature", 0))

    if provider == "deepseek":
        if not os.getenv("DEEPSEEK_API_KEY"):
            raise SystemExit(
                "[llm] 没找到 DEEPSEEK_API_KEY。请把 .env.example 复制为 .env，"
                "在里面填上 DEEPSEEK_API_KEY=sk-xxx。"
            )
        # 放在函数里 import：只用 Ollama 时，不装 langchain-deepseek 也能跑
        from langchain_deepseek import ChatDeepSeek

        return ChatDeepSeek(
            model=cfg["deepseek"]["model"], temperature=temperature, **overrides
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=cfg["ollama"]["model"],
            base_url=os.getenv("OLLAMA_BASE_URL") or cfg["ollama"].get("base_url", "http://localhost:11434"),
            temperature=temperature,
            reasoning=False,  # 关掉思考过程，不用再手动切 </think>
            **overrides,
        )

    raise SystemExit(f"[llm] 不认识的 provider: {provider!r}，可选 deepseek / ollama。")


@lru_cache
def get_embeddings() -> Embeddings:
    """把文字变成一串数字（向量）的模型，知识库检索要用。"""
    cfg = load_config().get("embedding")
    if not cfg:
        raise SystemExit(f"[llm] {CONFIG_PATH} 里缺少 embedding: 配置段。")
    from langchain_ollama import OllamaEmbeddings

    # Docker 里访问宿主机的 Ollama 要换地址，用环境变量 OLLAMA_BASE_URL 覆盖配置
    base_url = os.getenv("OLLAMA_BASE_URL") or cfg.get("base_url", "http://localhost:11434")
    return OllamaEmbeddings(model=cfg["model"], base_url=base_url)


if __name__ == "__main__":
    # 自检：python -m common.llm
    llm = get_llm()
    print(f"provider={load_llm_config()['provider']}, model={type(llm).__name__}")
    print(llm.invoke("用一句话介绍你自己").content)

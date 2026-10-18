"""Day3 用的两个简单工具。"""

from langchain_core.tools import tool
from day1_optimized_v5 import search_docs
from langgraph.types import interrupt


@tool
def get_weather(city: str) -> str:
    """查询某个城市今天的天气。"""
    fake = {"北京": "晴，25度", "上海": "小雨，22度"}
    return fake.get(city, "没有这个城市的数据")

@tool
def add(a: int, b: int) -> int:
    """计算两个整数的和。"""
    return a + b

@tool
def multiply(a:int , b:int) -> int:
    """计算两个整数乘积"""
    return a*b

@tool
def web_search(query:str) ->str:
    """联网搜索最新信息。遇到不知道或需要实时数据的问题时使用。"""
    return search_docs(query)

@tool
def send_email(to:str,content:str) -> str:
    """发送邮件"""
    decision = interrupt(f"确认发送给 {to}：{content}")
    if decision =="同意":
        return "邮件已发送"
    return "用户拒绝发送"

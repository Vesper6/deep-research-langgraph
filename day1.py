from typing import TypedDict
from langgraph.graph import StateGraph, START, END
import requests
class State(TypedDict):
    question:str
    plan:str
    context:str
    answer:str
    prompt:str


def make_plan(state:State) -> dict:
    return {"plan":"先找资料，再回答"}

def find_context(state:State) -> dict:
    return {"context":"这是一份假的资料"}


def write_answer(state:State) -> dict:
    return {"answer":"这是一份最终回答"}

def make_prompt(state:State)  -> dict:
    return{"prompt":"提示词"+ state['question']}

def call_llm(state:State) -> dict:
    r = requests.post("http://localhost:11434/api/generate", json={"model": "openbmb/minicpm-v4.6:latest", "prompt": state['prompt'], "stream": False})
    data = r.json()
    return {"answer": data["response"]}

def build_graph():
    graph=StateGraph(State)

    graph.add_node("make_plan",make_plan)
    graph.add_node("find_context",find_context)
    graph.add_node("write_answer",write_answer)
    graph.add_node("make_prompt",make_prompt)
    graph.add_node("call_llm",call_llm)

    graph.add_edge(START,"make_plan")
    graph.add_edge("make_plan","make_prompt")
    graph.add_edge("make_prompt","find_context")
    graph.add_edge("find_context","call_llm")
    graph.add_edge("call_llm",END   )
    #graph.add_edge("write_answer",END )

    return graph.compile()

if __name__ == "__main__":
    app =build_graph()

    result =app.invoke({"question":"这是一个问题",
                "plan":"",
                "context":"",
                "answer":"",
                "prompt":""})

    print(result)
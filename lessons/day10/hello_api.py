"""Day10 热身：最小的网页接口。运行：uvicorn lessons.day10.hello_api:app --reload"""

from fastapi import FastAPI

app = FastAPI()


@app.get("/hello")
def hello(name: str):
    return {"msg": f"你好，{name}"}

@app.get("/bye")
def bye(name:str):
    return {"msg":f"再见，{name}"}

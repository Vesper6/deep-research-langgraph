FROM python:3.13-slim

WORKDIR /app

# 先只复制依赖清单再安装：代码改了但依赖没变时，这一层能用缓存，重新构建很快
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir -p data

EXPOSE 8000
CMD ["uvicorn", "deep_research.api:app", "--host", "0.0.0.0", "--port", "8000"]

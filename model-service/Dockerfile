# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目
COPY . .

EXPOSE 5000

# 生产启动：4 workers + uvloop + httptools
CMD ["uvicorn","app:app","--host","0.0.0.0","--port","5000","--workers","4","--loop","uvloop","--http","httptools","--timeout-keep-alive","60"]

FROM python:3.9-slim-buster

WORKDIR /app

# 修复过期源并添加必要依赖 + ca 证书支持
RUN sed -i \
    -e 's|deb.debian.org/debian|archive.debian.org/debian|g' \
    -e 's|security.debian.org/debian-security|archive.debian.org/debian-security|g' \
    /etc/apt/sources.list && \
    echo 'Acquire::Check-Valid-Until "false";' > /etc/apt/apt.conf.d/99no-check-valid-until && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      libgl1-mesa-glx \
      curl \
      ca-certificates && \
    update-ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# 推荐：使用 Hugging Face 国内镜像（如需）
ENV HF_ENDPOINT=https://hf-mirror.com

# 安装 Python 依赖
COPY requirements.txt .
RUN pip install -i https://mirrors.aliyun.com/pypi/simple --timeout 120 -r requirements.txt

COPY . .

EXPOSE 5000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000", "--workers", "2", "--loop", "uvloop"]

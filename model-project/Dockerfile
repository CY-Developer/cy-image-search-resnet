FROM python:3.9-slim-buster

# ========== 系统依赖 ==========
RUN sed -i \
    -e 's|deb.debian.org/debian|archive.debian.org/debian|g' \
    -e 's|security.debian.org/debian-security|archive.debian.org/debian-security|g' \
    /etc/apt/sources.list && \
    echo 'Acquire::Check-Valid-Until "false";' > /etc/apt/apt.conf.d/99no-check-valid-until && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      libgl1-mesa-glx      \
      git \
      && rm -rf /var/lib/apt/lists/*

# ========== Python 依赖 ==========
COPY requirements.txt .
RUN pip install -i https://mirrors.aliyun.com/pypi/simple --timeout 120 -r requirements.txt

# 如果 requirements.txt 没写 torch/opencv，手动补：
#   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
#   pip install opencv-python-headless Pillow tqdm

# ========== 复制项目 ==========
WORKDIR /workspace
COPY . .

# 默认进入 bash；训练时手动执行
CMD ["/bin/bash"]




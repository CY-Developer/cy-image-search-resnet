FROM python:3.10-slim

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 仅训练所需库（不装浏览器）
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx git \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/model-project

# 安装依赖（requirements.txt 里已加 pandas）
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




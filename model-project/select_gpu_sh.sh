# 0) 进入项目目录，并优先启用脚本创建的虚拟环境（不会报错就忽略）
cd /path/to/你的项目根目录
source .venv/bin/activate 2>/dev/null || true

# 1) 确认宿主已挂载 GPU（这一条最关键）
nvidia-smi || echo "nvidia-smi 不存在（说明没挂 GPU 或镜像不含驱动库）"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<未设置>}"
ls /dev/nvidia* 2>/dev/null | wc -l

# 2) 看看当前 Python 里 torch 的真实状态
python - <<'PY'
import os, platform, sys
print("python:", sys.version)
try:
    import torch
    print("torch:", torch.__version__, "built_with_cuda:", torch.version.cuda)
    print("torch.cuda.is_available():", torch.cuda.is_available())
    print("device_count:", torch.cuda.device_count())
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    if torch.cuda.is_available():
        print("gpu0:", torch.cuda.get_device_name(0))
except Exception as e:
    print("import torch FAILED:", e)
PY

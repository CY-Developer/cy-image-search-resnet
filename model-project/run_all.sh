#!/usr/bin/env bash
# === run_all.sh (GPU 强制版，可直接替换) ===
set -euo pipefail
cd "$(dirname "$0")"

# ===== 0) GPU 绑定（默认 0 号卡；想换卡：CUDA_VISIBLE_DEVICES=1 bash run_all.sh）
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# 确保有 GPU（容器要有 nvidia-smi）
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[ERR] nvidia-smi 不可用：当前环境/镜像没挂 GPU/runtime" >&2
  exit 2
fi
nvidia-smi || true

# ===== 1) 路径配置（保持你的原始约定）
PROJECT_DIR="$(pwd)"                 # 项目根
IMG_ROOT="$PROJECT_DIR"              # ★你的图片根目录（务必与 CSV 相对路径对上）
CDIR="$PROJECT_DIR/cvs"              # CSV 目录
ODIR="$PROJECT_DIR/out"              # 输出目录

# ===== 2) Python venv + 依赖（自适配 GPU 版 PyTorch）
if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -V
pip install -U pip wheel
# 先装项目依赖（若有）
[[ -f requirements.txt ]] && pip install -r requirements.txt || true
pip install -U pandas

# 先检测现有 torch 是否已是 GPU 可用；不可用再按宿主 CUDA 安装对应轮子
set +e
python - <<'PY'
import os, sys
try:
    import torch
    print(f"[CHK] torch={torch.__version__} cuda_build={getattr(torch.version,'cuda',None)} cuda_avail={torch.cuda.is_available()}")
    print(f"[CHK] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    if torch.cuda.is_available():
        print("[CHK] device0:", torch.cuda.get_device_name(0))
        sys.exit(0)
    else:
        sys.exit(1)
except Exception as e:
    print("[CHK] import torch failed:", e)
    sys.exit(2)
PY
rc=$?
set -e

if [[ $rc -ne 0 ]]; then
  CUDA_VER=$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9]\+\.[0-9]\+\).*/\1/p' | head -n1)
  IDX=$([[ "$CUDA_VER" == 12.* ]] && echo cu121 || echo cu118)
  echo "[INFO] Installing PyTorch for $IDX (host CUDA=$CUDA_VER)"
  pip uninstall -y torch torchvision torchaudio >/dev/null 2>&1 || true
  pip install --index-url "https://download.pytorch.org/whl/$IDX" torch torchvision
fi

# 最终 GPU 断言（失败直接退出，不再跑 CPU）
python - <<'PY'
import torch, sys
assert torch.cuda.is_available(), "CUDA 不可用，请在算力平台选择含GPU的环境/镜像"
print("[GPU OK]", torch.cuda.get_device_name(0))
PY

# ===== 3) （可选）统一把 4 个 CSV 转 utf-8-sig，规避编码问题
python - <<'PY'
import os, io
def reencode(p):
    for enc in ("utf-8-sig","gb18030","utf-8"):
        try:
            with open(p, "r", encoding=enc, newline="") as f:
                s=f.read()
            with open(p, "w", encoding="utf-8-sig", newline="") as f:
                f.write(s)
            print(f"[ENCODE] {p} -> utf-8-sig (src={enc})")
            return
        except Exception:
            pass
    raise RuntimeError(f"无法解码: {p}")
base="cvs"
for fn in ["dummy_stage_one_train.csv","dummy_stage_one_verification.csv",
           "dummy_stage_two_train.csv","dummy_stage_two_verification.csv"]:
    p=os.path.join(base,fn)
    if os.path.exists(p): reencode(p)
PY

# ===== 4) Stage-1：Triplet-only 打底（保持你原参数 & 明确 --device cuda）
python -u "$PROJECT_DIR/train.py" \
  --csv_path         "$CDIR/dummy_stage_one_train.csv" \
  --val_csv_path     "$CDIR/dummy_stage_one_verification.csv" \
  --image_root       "$IMG_ROOT" \
  --mask_col         mask_path \
  --wm_col           wm_image_path \
  --epochs           30 \
  --batch_p          16 \
  --batch_k          12 \
  --lambda_cat       0.0 \
  --lambda_wm        0.0 \
  --lambda_inv       0.0 \
  --prefer_s01_ratio 0.0 \
  --num_workers      12 \
  --device           cuda \
  --use_aug \
  --use_cosine_lr \
  --lr               3e-4 \
  --outdir           "$ODIR/s1_e40_triplet_gpu"

# ===== 5) Stage-2：水印不变性优先（生产分布友好）
GM=""
if [[ -f "$CDIR/new_watermark.png" ]]; then
  GM="--global_watermark_path $CDIR/new_watermark.png"
  echo "[INFO] 使用全局水印: $CDIR/new_watermark.png"
else
  echo "[INFO] 未找到全局水印 PNG，跳过 --global_watermark_path（可选）"
fi

python - <<'PY'
import os, pandas as pd
base = os.environ.get("CDIR", "cvs")
fix = ["dummy_stage_two_train.csv", "dummy_stage_two_verification.csv"]
for fn in fix:
    p = os.path.join(base, fn)
    try:
        df = pd.read_csv(p)
    except Exception as e:
        print(f"[CSV FIX] skip {p}: {e}")
        continue
    # 统一类别
    df["category"] = (
        df["category"]
        .astype(str)
        .replace({"nan":"watch","NaN":"watch","None":"watch","":"watch"})
        .fillna("watch")
    )
    df.to_csv(p, index=False)
    print("[CSV FIX]", p, "unique categories =", df["category"].nunique(), df["category"].unique()[:5])
PY

python -u "$PROJECT_DIR/train.py" \
  --csv_path         "$CDIR/dummy_stage_two_train.csv" \
  --val_csv_path     "$CDIR/dummy_stage_two_verification.csv" \
  --image_root       "$IMG_ROOT" \
  --mask_col         mask_path \
  --wm_col           wm_image_path \
  $GM \
  --epochs           70 \
  --batch_p          16 \
  --batch_k          12 \
  --lambda_cat       0.0 \
  --lambda_wm        0.1 \
  --lambda_inv       0.40 \
  --prefer_s01_ratio 0.7 \
  --num_workers      12 \
  --device           cuda \
  --use_aug \
  --use_cosine_lr \
  --lr               2e-4 \
  --resume           "$ODIR/s1_e40_triplet_gpu/best.pth" \
  --outdir           "$ODIR/s2_e70_wminv_prod_gpu"

# ===== 6) 评测（同一验证集 A/B，对比 Recall@1）
python "$PROJECT_DIR/quick_report.py" \
  --ckpt       "$ODIR/s1_e40_triplet_gpu/best.pth" \
  --val_csv    "$CDIR/dummy_stage_two_verification.csv" \
  --image_root "$IMG_ROOT" \
  --device     cuda \
  --batch_size 128 \
  --center_crop

python "$PROJECT_DIR/quick_report.py" \
  --ckpt       "$ODIR/s2_e70_wminv_prod_gpu/best.pth" \
  --val_csv    "$CDIR/dummy_stage_two_verification.csv" \
  --image_root "$IMG_ROOT" \
  --device cuda \
  --batch_size 128 \
  --center_crop \
  --model_entry MultiTaskResNet \
  --entry_args '{"embed_dim":512,"num_categories":1,"pretrained":false,"l2_norm":true}'

echo "[DONE] 训练与评测完成。权重在："
echo " - Stage-1: $ODIR/s1_e40_triplet_gpu/best.pth"
echo " - Stage-2: $ODIR/s2_e70_wminv_prod_gpu/best.pth"

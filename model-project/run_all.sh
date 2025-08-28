cat > run_all.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

# ====== 0) 路径配置（按需改：只需确认 IMG_ROOT） ======
PROJECT_DIR="$(pwd)"                 # 当前目录作为项目根
IMG_ROOT="$PROJECT_DIR"              # ★你的图片根目录（务必与 CSV 相对路径对上）
CDIR="$PROJECT_DIR/cvs"              # CSV 目录
ODIR="$PROJECT_DIR/out"              # 输出目录

# ====== 1) 环境准备：虚拟环境 + GPU PyTorch（cu121 适配 4090） ======
python -m venv .venv
source .venv/bin/activate
pip install -U pip wheel
pip install -r requirements.txt || true
pip install --index-url https://download.pytorch.org/whl/cu121 --upgrade "torch==2.8.0" "torchvision==0.23.0"

# 简单检查 GPU
python - <<'PY'
import torch, sys
print(f"[CUDA] available={torch.cuda.is_available()}")
print(f"[CUDA] name={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
assert torch.cuda.is_available(), "CUDA 不可用，请在算力平台选择含GPU的环境/镜像"
PY

# ====== 2) （可选但推荐）统一把 4 个 CSV 转为 utf-8-sig，规避编码问题 ======
python - <<'PY'
import os, csv, io, codecs, shutil
def reencode(p):
    for enc in ("utf-8-sig","gb18030"):
        try:
            with open(p, "r", encoding=enc, newline="") as f:
                s=f.read()
            with open(p, "w", encoding="utf-8-sig", newline="") as f:
                f.write(s)
            print(f"[ENCODE] {p} -> utf-8-sig (src={enc})"); return
        except Exception: pass
    raise RuntimeError(f"无法解码: {p}")
base="cvs"
for fn in ["dummy_stage_one_train.csv","dummy_stage_one_verification.csv",
           "dummy_stage_two_train.csv","dummy_stage_two_verification.csv"]:
    p=os.path.join(base,fn)
    if os.path.exists(p): reencode(p)
PY

# ====== 3) Stage-1：Triplet-only 打底 ======
python -u "$PROJECT_DIR/train.py" \
  --csv_path         "$CDIR/dummy_stage_one_train.csv" \
  --val_csv_path     "$CDIR/dummy_stage_one_verification.csv" \
  --image_root       "$IMG_ROOT" \
  --mask_col         mask_path \
  --wm_col           wm_image_path \
  --epochs           40 \
  --batch_p          16 \
  --batch_k          8 \
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

# ====== 4) Stage-2：水印不变性优先（生产分布友好） ======
GM=""
if [[ -f "$CDIR/new_watermark.png" ]]; then
  GM="--global_watermark_path $CDIR/new_watermark.png"
  echo "[INFO] 使用全局水印: $CDIR/new_watermark.png"
else
  echo "[INFO] 未找到全局水印 PNG，跳过 --global_watermark_path（可选）"
fi

python -u "$PROJECT_DIR/train.py" \
  --csv_path         "$CDIR/dummy_stage_two_train.csv" \
  --val_csv_path     "$CDIR/dummy_stage_two_verification.csv" \
  --image_root       "$IMG_ROOT" \
  --mask_col         mask_path \
  --wm_col           wm_image_path \
  $GM \
  --epochs           70 \
  --batch_p          16 \
  --batch_k          8 \
  --lambda_cat       0.0 \
  --lambda_wm        0.1 \
  --lambda_inv       0.45 \
  --prefer_s01_ratio 0.7 \
  --num_workers      12 \
  --device           cuda \
  --use_aug \
  --use_cosine_lr \
  --lr               2e-4 \
  --resume           "$ODIR/s1_e40_triplet_gpu/best.pth" \
  --outdir           "$ODIR/s2_e70_wminv_prod_gpu"

# ====== 5) 评测（同一验证集 A/B，对比 Recall@1） ======
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
  --device     cuda \
  --batch_size 128 \
  --center_crop

echo "[DONE] 训练与评测完成。权重在："
echo " - Stage-1: $ODIR/s1_e40_triplet_gpu/best.pth"
echo " - Stage-2: $ODIR/s2_e70_wminv_prod_gpu/best.pth"
EOF

bash run_all.sh

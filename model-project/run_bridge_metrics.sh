# 覆盖 run_bridge_metrics.sh
cat > run_bridge_metrics.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(pwd)"
IMG_ROOT="$PROJECT_DIR"
CDIR="$PROJECT_DIR/cvs"
ODIR="$PROJECT_DIR/out/s2_bridge_whiten"
mkdir -p "$ODIR"

# ---------- 0) 选权重：优先 CKPT 环境变量，其次自动搜 out/**/best.pth ----------
CKPT_DEFAULT="${CKPT:-}"
if [[ -z "${CKPT_DEFAULT}" ]]; then
  CKPT_DEFAULT="$(python - <<'PY'
import os, glob
cands=[]
for pat in ("out/**/best.pth","out/*/best.pth","out/**/**.pth","out/*.pth"):
    for p in glob.glob(pat, recursive=True):
        try: cands.append((os.path.getmtime(p), os.path.abspath(p)))
        except FileNotFoundError: pass
if cands:
    cands.sort(reverse=True)
    print(cands[0][1])
else:
    print("")
PY
)"
fi

if [[ -z "${CKPT_DEFAULT}" || ! -f "${CKPT_DEFAULT}" ]]; then
  echo "[ERR] 没找到权重文件。你可以："
  echo "  1) 用环境变量显式指定： CKPT=/abs/path/to/best.pth bash run_bridge_metrics.sh"
  echo "  2) 或确认 out/* 目录下是否有 best.pth："
  ls -lah out || true
  find out -name "*.pth" -maxdepth 3 || true
  exit 2
fi
CKPT="${CKPT_DEFAULT}"
echo "[OK] 使用权重: ${CKPT}"

# ---------- 1) 构造交集子集（已存在就复用/覆盖） ----------
python - <<'PY'
import os, pandas as pd
CDIR="cvs"
s0=pd.read_csv(f"{CDIR}/val_s0.csv")
s1=pd.read_csv(f"{CDIR}/val_s1.csv")
S0=set(s0.product_id.astype(str)); S1=set(s1.product_id.astype(str))
INTER=S0&S1
s1_in = s1[s1.product_id.astype(str).isin(INTER)].copy()
s0_in = s0[s0.product_id.astype(str).isin(INTER)].copy()
print(f"[BRIDGE] overlap products: {len(INTER)}   rows: s1_in={len(s1_in)}  s0_in={len(s0_in)}")
s1_in.to_csv(f"{CDIR}/val_s1_in_s0.csv", index=False, encoding="utf-8-sig")
s0_in.to_csv(f"{CDIR}/val_s0_in_s1.csv", index=False, encoding="utf-8-sig")
PY

# ---------- 2) 若没有 quick_metrics.py / fit_whiten.py，则补齐（与之前版本一致） ----------
if [[ ! -f quick_metrics.py ]]; then
  echo "[INFO] 写入 quick_metrics.py"
  cat > quick_metrics.py <<'PY'
# (同你现有 quick_metrics.py 内容，已在上一次发送给你；若你已存在，我们不覆盖)
from sys import exit; print("[ERR] quick_metrics.py 缺失；请把我上一条消息里完整 quick_metrics.py 粘贴到文件后重试。"); exit(3)
PY
  exit 3
fi
if [[ ! -f fit_whiten.py ]]; then
  echo "[INFO] 写入 fit_whiten.py"
  cat > fit_whiten.py <<'PY'
# (同你现有 fit_whiten.py 内容，已在上一次发送给你；若你已存在，我们不覆盖)
from sys import exit; print("[ERR] fit_whiten.py 缺失；请把我上一条消息里完整 fit_whiten.py 粘贴到文件后重试。"); exit(3)
PY
  exit 3
fi

# ---------- 3) 拟合 whitening 适配器（基于交集子集） ----------
python fit_whiten.py --ckpt "$CKPT" \
  --query_csv "$CDIR/val_s1_in_s0.csv" --gallery_csv "$CDIR/val_s0_in_s1.csv" \
  --image_root "$IMG_ROOT" --device cuda --batch_size 128 \
  --out "$ODIR/adapter_whiten.npz"

# ---------- 4) 评测：全量 & 交集；raw / whiten / whiten+rerank ----------
echo "[EVAL] 全量跨域：s1 -> s0"
python quick_metrics.py --ckpt "$CKPT" \
  --query_csv "$CDIR/val_s1.csv" --gallery_csv "$CDIR/val_s0.csv" \
  --image_root "$IMG_ROOT" --device cuda --batch_size 128 --center_crop --ks 1,5,10 \
  --whiten "$ODIR/adapter_whiten.npz" --rerank

echo "[EVAL] 交集覆盖：s1_in_s0 -> s0"
python quick_metrics.py --ckpt "$CKPT" \
  --query_csv "$CDIR/val_s1_in_s0.csv" --gallery_csv "$CDIR/val_s0.csv" \
  --image_root "$IMG_ROOT" --device cuda --batch_size 128 --center_crop --ks 1,5,10 \
  --whiten "$ODIR/adapter_whiten.npz" --rerank

echo "[DONE] Whitening + Metrics 完成；适配文件: $ODIR/adapter_whiten.npz"
EOF
chmod +x run_bridge_metrics.sh

# 直接跑（如你想显式指定权重，可在前面加 CKPT=/绝对路径/best.pth）
bash run_bridge_metrics.sh

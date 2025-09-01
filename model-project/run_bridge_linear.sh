#!/usr/bin/env bash
set -euo pipefail
ROOT="$(pwd)"
CDIR="$ROOT/cvs"
ODIR="$ROOT/out/bridge_linear"
mkdir -p "$ODIR"

# 1) 选择权重：支持 CKPT=... 覆盖；否则自动搜 out/**/best.pth
CKPT="${CKPT:-$(python - <<'PY'
import os, glob
cands=[]
for pat in ("out/**/best.pth","out/*/best.pth","out/**/*.pth","out/*.pth"):
    for p in glob.glob(pat, recursive=True):
        try: cands.append((os.path.getmtime(p), os.path.abspath(p)))
        except: pass
cands.sort(reverse=True)
print(cands[0][1] if cands else "")
PY
)}"
if [[ -z "$CKPT" || ! -f "$CKPT" ]]; then
  echo "[ERR] 没找到 .pth，请用 CKPT=/abs/path/best.pth bash run_bridge_linear.sh"; exit 2
fi
echo "[OK] 使用权重: $CKPT"

# 2) 用你已经有的 val_s1/val_s0 的交集来拟合线性桥接
python - <<'PY'
import pandas as pd
s0=pd.read_csv("cvs/val_s0.csv"); s1=pd.read_csv("cvs/val_s1.csv")
S0=set(s0.product_id.astype(str)); S1=set(s1.product_id.astype(str))
inter=S0&S1
s1_in=s1[s1.product_id.astype(str).isin(inter)]
s0_in=s0[s0.product_id.astype(str).isin(inter)]
s1_in.to_csv("cvs/val_s1_in_s0.csv",index=False,encoding="utf-8-sig")
s0_in.to_csv("cvs/val_s0_in_s1.csv",index=False,encoding="utf-8-sig")
print(f"[INTER] products={len(inter)}  rows: s1_in={len(s1_in)} s0_in={len(s0_in)}")
PY

# 3) 训练线性桥接（极快）
python fit_linear_bridge.py --ckpt "$CKPT" \
  --s1_csv "$CDIR/val_s1_in_s0.csv" --s0_csv "$CDIR/val_s0_in_s1.csv" \
  --image_root "$ROOT" --device cuda --batch_size 128 \
  --out "$ODIR/adapter_linear.npz"

# 4) 评测：全量 + 交集，raw / bridge / bridge+rerank
echo "[EVAL] s1 -> s0 (全量)"
python quick_metrics_dual.py --ckpt "$CKPT" \
  --query_csv "$CDIR/val_s1.csv" --gallery_csv "$CDIR/val_s0.csv" \
  --image_root "$ROOT" --device cuda --batch_size 128 --center_crop --ks 1,5,10 \
  --bridge "$ODIR/adapter_linear.npz" --rerank

echo "[EVAL] s1_in_s0 -> s0 (交集覆盖)"
python quick_metrics_dual.py --ckpt "$CKPT" \
  --query_csv "$CDIR/val_s1_in_s0.csv" --gallery_csv "$CDIR/val_s0.csv" \
  --image_root "$ROOT" --device cuda --batch_size 128 --center_crop --ks 1,5,10 \
  --bridge "$ODIR/adapter_linear.npz" --rerank

echo "[DONE] 线性桥接训练+评测完成：$ODIR/adapter_linear.npz"

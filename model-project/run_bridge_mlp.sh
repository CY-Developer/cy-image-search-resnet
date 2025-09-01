cd /home/featurize/app/model-project

cat > run_bridge_mlp.sh <<'SH'
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

CKPT="${CKPT:-$(find out -type f -path '*/s2_*/*' -name best.pth -printf '%T@ %p\n' | sort -nr | awk 'NR==1{print $2}')}"

echo "[CKPT]" "$CKPT"
if [[ -z "${CKPT}" || ! -f "${CKPT}" ]]; then
  echo "[ERR] 找不到 Stage-2 权重: ${CKPT:-<空>}"
  echo "用法：CKPT=/abs/path/best.pth bash run_bridge_mlp.sh"; exit 2;
fi

python - <<'PY'
import os, csv, numpy as np, torch, torch.nn as nn, torch.optim as optim
import torch.nn.functional as F
from collections import defaultdict
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from quick_report_cross import load_model_from_ckpt

ROOT="."; DEV="cuda"
torch.set_float32_matmul_precision("high")

def l2n(x): return F.normalize(x, dim=1)

def ensure_overlap_csv():
    import pandas as pd
    base="cvs"
    s0=pd.read_csv(os.path.join(base,"val_s0.csv"))
    s1=pd.read_csv(os.path.join(base,"val_s1.csv"))
    S0=set(s0.product_id.astype(str)); S1=set(s1.product_id.astype(str))
    common=S0 & S1
    s1_in=s1[s1.product_id.astype(str).isin(common)]
    s0_in=s0[s0.product_id.astype(str).isin(common)]
    os.makedirs(base,exist_ok=True)
    s1_in.to_csv(os.path.join(base,"val_s1_in_s0.csv"),index=False,encoding="utf-8-sig")
    s0_in.to_csv(os.path.join(base,"val_s0_in_s1.csv"),index=False,encoding="utf-8-sig")
    print(f"[OVERLAP] products={len(common)}  rows: s1_in={len(s1_in)}  s0_in={len(s0_in)}")

IMAGENET_MEAN=(0.485,0.456,0.406)
IMAGENET_STD =(0.229,0.224,0.225)
tx = T.Compose([
    T.Resize(256, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

class CSVImageDataset(Dataset):
    def __init__(self, csv_path:str, image_root:str):
        self.root=image_root
        rows=[]
        with open(csv_path, newline='', encoding='utf-8-sig') as f:
            rd=csv.DictReader(f)
            for r in rd:
                p=r.get('image_path','').strip()
                pid=str(r.get('product_id','')).strip()
                if not p or not pid: continue
                if os.path.isabs(p):
                    try: p=os.path.relpath(p, image_root)
                    except: pass
                full=os.path.normpath(os.path.join(image_root, p))
                if os.path.isfile(full):
                    rows.append((p, pid))
        if not rows:
            raise RuntimeError(f"[EVAL] 空集: {csv_path}")
        self.rows=rows
        self.pids=[pid for _,pid in rows]
        print(f"[DATASET] {csv_path}: rows={len(rows)} uniq_pids={len(set(self.pids))}")

    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        rel, pid = self.rows[i]
        from PIL import Image
        img=Image.open(os.path.join(self.root, rel)).convert('RGB')
        return tx(img), pid

@torch.no_grad()
def extract_feats_generic(model, ds, device="cuda", bs=256):
    dl=DataLoader(ds, batch_size=bs, shuffle=False, num_workers=2, pin_memory=True)
    feats=[]; pids=[]
    model.eval()
    for ims, pid in dl:
        ims = ims.to(device, non_blocking=True)
        out = model(ims)
        if isinstance(out, (list,tuple)): out=out[0]
        if isinstance(out, dict):
            for k in ('feats','features','emb','embedding','pool','penult'):
                if k in out: out=out[k]; break
        if out.ndim==4:
            out = torch.nn.functional.adaptive_avg_pool2d(out,1).squeeze(-1).squeeze(-1)
        if out.ndim!=2:
            out = out.reshape(out.size(0), -1)
        feats.append(out.detach().float().cpu())
        pids += list(pid)
    feats = torch.cat(feats, 0)
    return l2n(feats), np.array([str(x) for x in pids])

def centroids(feats, pids):
    idxs=defaultdict(list)
    for i,p in enumerate(pids): idxs[p].append(i)
    keys=sorted(idxs.keys())
    outs=[feats[idxs[k]].mean(0,keepdim=True) for k in keys]
    return l2n(torch.cat(outs,0)), np.array(keys)

def recall_map(qf, qp, gf, gp, ks=(1,5,10)):
    import numpy as np
    val_mask = np.isin(qp, gp)
    qf, qp = qf[val_mask], qp[val_mask]
    if len(qp)==0:
        return {f"R@{k}":"0.00%" for k in ks} | {"mAP":"0.00%","_valid_queries":0}
    q = qf / np.linalg.norm(qf,axis=1,keepdims=True)
    g = gf / np.linalg.norm(gf,axis=1,keepdims=True)
    sims = q @ g.T
    R={k:0 for k in ks}; ap_sum=0.0
    for i in range(len(qp)):
        idx=np.argsort(-sims[i]); gp_sorted=gp[idx]
        hits=(gp_sorted==qp[i]).astype(np.int32)
        pos=np.where(hits==1)[0]
        if pos.size==0: continue
        r0=pos[0]
        for k in ks:
            if r0<k: R[k]+=1
        c=np.cumsum(hits); ap=(c[pos]/(pos+1)).mean(); ap_sum+=ap
    n=max(1,len(qp))
    out={f"R@{k}":f"{R[k]/n*100:.2f}%" for k in ks}
    out["mAP"]=f"{ap_sum/n*100:.2f}%"; out["_valid_queries"]=len(qp)
    return out

CKPT=os.environ["CKPT"]
model = load_model_from_ckpt(CKPT, DEV).eval()

ensure_overlap_csv()

ds_q = CSVImageDataset("cvs/val_s1_in_s0.csv", ROOT)
ds_g = CSVImageDataset("cvs/val_s0_in_s1.csv", ROOT)
f_q, p_q = extract_feats_generic(model, ds_q, DEV, 256)
f_g, p_g = extract_feats_generic(model, ds_g, DEV, 256)

cq, pids_q = centroids(f_q, p_q)
cg, pids_g = centroids(f_g, p_g)
common = np.array(sorted(set(pids_q) & set(pids_g)))
sel_q = torch.stack([cq[ np.where(pids_q==pid)[0][0] ] for pid in common], dim=0)
sel_g = torch.stack([cg[ np.where(pids_g==pid)[0][0] ] for pid in common], dim=0)
pairs = sel_q.shape[0]
print(f"[TRAIN DATA] pairs={pairs}  dim={sel_q.shape[1]}")

mlp = nn.Sequential(
    nn.Linear(sel_q.shape[1], 1024, bias=True),
    nn.ReLU(inplace=True),
    nn.Linear(1024, sel_q.shape[1], bias=True)
).to(DEV)
opt = optim.AdamW(mlp.parameters(), lr=1e-3, weight_decay=1e-4)

for e in range(8):
    mlp.train()
    idx = torch.randperm(pairs)   # <<< 索引放 CPU，避免设备不匹配
    x = sel_q[idx].to(DEV)
    y = sel_g[idx].to(DEV)
    z = l2n(mlp(x))
    loss = 1 - (z*y).sum(1).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    print({"epoch":e+1, "loss": float(loss)})

os.makedirs("out/bridge_mlp", exist_ok=True)
torch.save(mlp.state_dict(), "out/bridge_mlp/adapter_mlp.pt")
print("[SAVE] out/bridge_mlp/adapter_mlp.pt")

def eval_split(q_csv, g_csv, title):
    qf, qp = extract_feats_generic(model, CSVImageDataset(q_csv, ROOT), DEV, 256)
    gf, gp = extract_feats_generic(model, CSVImageDataset(g_csv, ROOT), DEV, 256)
    raw = recall_map(qf.numpy(), qp, gf.numpy(), gp)
    with torch.no_grad():
        qf_mlp = l2n(mlp(qf.to(DEV))).cpu().numpy()
    brid = recall_map(qf_mlp, qp, gf.numpy(), gp)
    print(f"[EVAL] {title}")
    print("[METRIC][raw] ", raw)
    print("[METRIC][mlp] ", brid)

eval_split("cvs/val_s1.csv",       "cvs/val_s0.csv",       "s1 -> s0 (全量)")
eval_split("cvs/val_s1_in_s0.csv", "cvs/val_s0.csv",       "s1_in_s0 -> s0 (交集覆盖)")
PY
SH
chmod +x run_bridge_mlp.sh

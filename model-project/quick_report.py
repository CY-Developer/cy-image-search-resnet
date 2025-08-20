"""
quick_report.py
----------------
轻量自测脚本：在“CPU 小数据”场景下，验证你训练出来的 .pth 模型是否可用、向量是否分得开。
- 读取一个验证集 CSV（至少包含 image_path, product_id 两列）
- 可选使用每图 mask_path 或一张全局水印 PNG（alpha 通道）做 mask-gating
- 支持两类 checkpoint：
  1) torch.save(model, path)        # 直接保存整模
  2) torch.save({'state_dict':...}) # 仅保存权重（需能从 model.py 构建模型）
主要输出：
- 数据健康：图片数、产品数、每产品图片数分布、单图产品占比
- 模型可用性：Recall@1/5、类内/类间平均相似度、推理吞吐（CPU）
- 一致性：同一批前向两次的最大差异
- 保存报告：./quick_reports/report_YYYYMMDD_HHMMSS.md
用法示例：
python quick_report.py \
  --ckpt outputs/fan/best.pth \
  --val_csv data/val_fan.csv \
  --image_root /workspace \
  --device cpu \
  --batch_size 8 \
  --model_entry auto \
  --entry_args "{}" \
  --img_size 224 \
  --center_crop \
  --use_mask_gating \
  --global_watermark_path data/watermarks/wm.png \
  --alpha_threshold 0.5
"""
import argparse, os, time, importlib, inspect, json
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import numpy as np, pandas as pd
from PIL import Image
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from datetime import datetime

IMAGENET_MEAN=[0.485,0.456,0.406]; IMAGENET_STD=[0.229,0.224,0.225]
def log(s): print(s, flush=True)
def l2_normalize(x:torch.Tensor, eps:float=1e-12)->torch.Tensor: return x/(x.norm(p=2,dim=1,keepdim=True)+eps)
def try_import(m):
    try: return importlib.import_module(m)
    except Exception: return None

def read_global_mask(path:Optional[str], size:Optional[Tuple[int,int]], alpha_threshold:float):
    if not path or not os.path.isfile(path): return None
    img=Image.open(path).convert("RGBA"); A=img.split()[3]
    if size is not None: A=A.resize(size, resample=Image.BILINEAR)
    th=int(max(0,min(1,alpha_threshold))*255)
    return A.point(lambda p:255 if p>=th else 0)

class ValDataset(Dataset):
    def __init__(self,csv_path,image_root=None,img_size=224,center_crop=True,mask_col="mask_path",
                 use_mask_gating=False,global_mask_path=None,alpha_threshold=0.5):
        self.df=pd.read_csv(csv_path)
        if "image_path" not in self.df.columns or "product_id" not in self.df.columns:
            raise ValueError("CSV 必须包含列：image_path, product_id（可选：mask_path）")
        self.image_root=image_root; self.mask_col=mask_col
        self.use_mask_gating=use_mask_gating; self.alpha_threshold=alpha_threshold
        self.global_mask_path=global_mask_path
        self.tf=T.Compose([
            T.Resize(int(img_size*256/224)) if center_crop else T.Resize((img_size,img_size)),
            T.CenterCrop(img_size) if center_crop else (lambda x:x),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN,std=IMAGENET_STD),
        ])
        self.records=[]
        for _,r in self.df.iterrows():
            ip=str(r["image_path"]);
            if image_root and not os.path.isabs(ip): ip=os.path.join(image_root,ip)
            pid=str(r["product_id"]); mp=None
            if self.mask_col in self.df.columns and not pd.isna(r[self.mask_col]):
                mp=str(r[self.mask_col]);
                if image_root and not os.path.isabs(mp): mp=os.path.join(image_root,mp)
            self.records.append((ip,pid,mp))
    def __len__(self): return len(self.records)
    def _load_mask_tensor(self,mask_path,hw):
        H,W=hw; composed=None
        if mask_path and os.path.isfile(mask_path):
            try: composed=Image.open(mask_path).convert("L").resize((W,H),resample=Image.BILINEAR)
            except Exception: composed=None
        if self.global_mask_path:
            gm=read_global_mask(self.global_mask_path,(W,H),self.alpha_threshold)
            if gm is not None:
                composed=gm if composed is None else Image.fromarray(np.maximum(np.array(composed),np.array(gm)).astype(np.uint8))
        if composed is None: return None
        arr=np.array(composed).astype(np.float32)/255.0
        return torch.from_numpy(arr).unsqueeze(0)
    def __getitem__(self,idx):
        ipath,pid,mpath=self.records[idx]
        img=Image.open(ipath).convert("RGB"); x=self.tf(img)
        if self.use_mask_gating:
            mt=self._load_mask_tensor(mpath,(x.shape[1],x.shape[2]))
            if mt is not None: x=x*(1.0-mt)
        return x,pid,ipath

COMMON_ENTRIES=["create_model","build_model","get_model","Model","Net","MultiTaskModel","Backbone"]
def build_model(entry:str,cfg:Dict[str,Any]):
    mod=try_import("model")
    if mod is None: raise RuntimeError("未找到 model.py（请在项目根运行，或把根目录加到 PYTHONPATH）")
    if not hasattr(mod,entry): raise AttributeError(f"model.py 无入口 '{entry}'；可用 --model_entry auto")
    ctor=getattr(mod,entry)
    if inspect.isfunction(ctor):
        try: return ctor(**cfg)
        except TypeError: return ctor()
    elif inspect.isclass(ctor):
        try: return ctor(**cfg)
        except TypeError: return ctor()
    else: raise TypeError("入口既不是函数也不是类")
def auto_build(cfg):
    for e in COMMON_ENTRIES:
        try: return build_model(e,cfg)
        except Exception: continue
    raise RuntimeError("自动构建模型失败，请显式 --model_entry 并配合 --entry_args")
def load_checkpoint(ckpt_path,device,model_entry,entry_args):
    obj=torch.load(ckpt_path,map_location=device)
    if isinstance(obj,nn.Module): return obj.to(device).eval()
    if isinstance(obj,dict) and "model" in obj and isinstance(obj["model"],nn.Module): return obj["model"].to(device).eval()
    if isinstance(obj,dict):
        state=obj["state_dict"] if "state_dict" in obj else obj
        model=auto_build(entry_args) if model_entry=="auto" else build_model(model_entry,entry_args)
        missing,unexpected=model.load_state_dict(state,strict=False)
        if missing: print(f"[WARN] missing keys: {missing[:8]}{' ...' if len(missing)>8 else ''}")
        if unexpected: print(f"[WARN] unexpected keys: {unexpected[:8]}{' ...' if len(unexpected)>8 else ''}")
        return model.to(device).eval()
    raise RuntimeError("无法识别的 checkpoint 格式")

def forward_get_embedding(model,x):
    with torch.no_grad(): out=model(x)
    if isinstance(out,torch.Tensor): emb=out
    elif isinstance(out,(list,tuple)) and len(out)>0: emb=out[0]
    elif isinstance(out,dict):
        for k in ["emb","embedding","embeddings","feat","features"]:
            if k in out: emb=out[k]; break
        else: raise RuntimeError("字典输出中未找到 embedding 字段")
    else: raise RuntimeError("未知的模型输出类型")
    return emb.unsqueeze(0) if emb.dim()==1 else emb

def recall_at_k(emb,labels,ks=(1,5)):
    sim=emb@emb.T; n=sim.shape[0]; np.fill_diagonal(sim,-1.0)
    L=np.array(labels); recalls={}
    idx_topk=np.argpartition(-sim,kth=max(0,max(ks)-1),axis=1)[:,:max(ks)]
    for k in ks:
        correct=0
        for i in range(n):
            if (L[idx_topk[i,:k]]==L[i]).any(): correct+=1
        recalls[k]=correct/n if n>0 else 0.0
    return recalls
def intra_inter_similarity(emb,labels):
    n=emb.shape[0];
    if n<2: return float("nan"),float("nan")
    sim=emb@emb.T; np.fill_diagonal(sim,np.nan); L=np.array(labels)
    same=(L[:,None]==L[None,:]); diff=~same
    intra=np.nanmean(sim[same & ~np.eye(n,dtype=bool)]) if np.any(same) else float("nan")
    inter=np.nanmean(sim[diff]) if np.any(diff) else float("nan")
    return float(intra),float(inter)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--ckpt",required=True)
    ap.add_argument("--val_csv",required=True)
    ap.add_argument("--image_root",default="")
    ap.add_argument("--device",default="cpu",choices=["cpu","cuda","auto"])
    ap.add_argument("--batch_size",type=int,default=16)
    ap.add_argument("--num_workers",type=int,default=0)
    ap.add_argument("--img_size",type=int,default=224)
    ap.add_argument("--center_crop",action="store_true")
    ap.add_argument("--use_mask_gating",action="store_true")
    ap.add_argument("--global_watermark_path",default="")
    ap.add_argument("--alpha_threshold",type=float,default=0.5)
    ap.add_argument("--mask_col",default="mask_path")
    ap.add_argument("--model_entry",default="auto")
    ap.add_argument("--entry_args",default="{}")
    args=ap.parse_args()
    device="cuda" if (args.device=="auto" and torch.cuda.is_available()) else (args.device if args.device!="auto" else "cpu")
    log(f"[Device] {device}")
    ds=ValDataset(args.val_csv, args.image_root if args.image_root else None, args.img_size, args.center_crop,
                  args.mask_col, args.use_mask_gating, args.global_watermark_path if args.global_watermark_path else None, args.alpha_threshold)
    dl=DataLoader(ds,batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers,pin_memory=(device=="cuda"))
    df=pd.read_csv(args.val_csv)
    imgs_per_prod=df.groupby("product_id")["image_path"].count().values
    one_ratio=float((imgs_per_prod==1).sum()/max(1,len(imgs_per_prod))) if len(imgs_per_prod) else 0.0
    log(f"[Data] images={len(df)} products={df['product_id'].nunique()} imgs/prod (min/mean/median/max)=({imgs_per_prod.min() if len(imgs_per_prod) else 0}/{np.mean(imgs_per_prod) if len(imgs_per_prod) else 0:.2f}/{np.median(imgs_per_prod) if len(imgs_per_prod) else 0}/{imgs_per_prod.max() if len(imgs_per_prod) else 0}) single-prod-ratio={one_ratio:.2%}")
    try: entry_args=json.loads(args.entry_args or "{}")
    except Exception: entry_args={}
    model=load_checkpoint(args.ckpt,device,args.model_entry,entry_args).eval()
    embs=[]; labels=[]; paths=[]; diffs=[]; t0=time.time()
    for x,pids,ips in dl:
        x=x.to(device,non_blocking=True)
        with torch.no_grad():
            e1=forward_get_embedding(model,x); e2=forward_get_embedding(model,x)
        diffs.append(float((e1-e2).abs().max().detach().cpu().item()))
        embs.append(l2_normalize(e1).detach().cpu()); labels.extend([str(z) for z in pids]); paths.extend(list(ips))
    sec=max(1e-6,time.time()-t0); total=len(ds); ips=total/sec; max_diff=max(diffs) if diffs else 0.0
    E=torch.cat(embs,dim=0).numpy() if embs else np.zeros((0,1),dtype=np.float32)
    rec=recall_at_k(E,labels,ks=(1,5)) if len(E)>1 else {1:0.0,5:0.0}
    intra,inter=intra_inter_similarity(E,labels)
    outdir=Path("quick_reports"); outdir.mkdir(parents=True,exist_ok=True)
    rpt=outdir/f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    def ok(f): return "✅" if f else "⚠️"
    with rpt.open("w",encoding="utf-8") as f:
        f.write(f"# quick_report\n\n**Checkpoint**: `{args.ckpt}`\n\n## 数据健康\n")
        f.write(f"- 图片数：{len(df)}\n- 产品数：{df['product_id'].nunique()}\n")
        if len(imgs_per_prod): f.write(f"- 每产品图片数(min/mean/median/max)：{imgs_per_prod.min()}/{np.mean(imgs_per_prod):.2f}/{np.median(imgs_per_prod)}/{imgs_per_prod.max()}\n")
        f.write(f"- 单图产品占比：{one_ratio:.2%} {ok(one_ratio<=0.05)}\n\n## 模型可用性\n")
        f.write(f"- Recall@1：{rec.get(1,0.0):.4f}  {ok(rec.get(1,0.0)>=0.95)}\n- Recall@5：{rec.get(5,0.0):.4f}  {ok(rec.get(5,0.0)>=0.98)}\n")
        f.write(f"- 类内平均相似度：{intra:.4f}\n- 类间平均相似度：{inter:.4f}\n\n## 推理一致性与吞吐\n")
        f.write(f"- 同一批两次前向最大差异（应接近 0）：{max_diff:.6f}\n- 推理吞吐（图/秒，{device}）：{ips:.2f}\n\n## 运行参数\n")
        f.write(f"- 设备：{device}\n- 批大小：{args.batch_size}\n- 图像尺寸：{args.img_size}\n- CenterCrop：{args.center_crop}\n- 掩模 gating：{args.use_mask_gating}\n")
        f.write(f"- 全局水印 PNG：{args.global_watermark_path or '无'}\n- 掩模列：{args.mask_col}\n- model_entry：{args.model_entry}\n- entry_args：{entry_args}\n")
    log("\n======== SUMMARY ========")
    log(f"Recall@1={rec.get(1,0.0):.4f}  Recall@5={rec.get(5,0.0):.4f}  Intra={intra:.4f}  Inter={inter:.4f}")
    log(f"MaxDiff={max_diff:.6f}  Throughput={ips:.2f} img/s on {device}")
    log(f"Report saved to: {rpt}")
    log("=========================")

if __name__=="__main__": main()











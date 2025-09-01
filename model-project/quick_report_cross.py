#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, argparse, json, math, random
import pandas as pd
import numpy as np
from PIL import Image, ImageOps

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as M

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def build_transform(img_size=224, center_crop=False):
    t = [T.Resize(256)]
    t.append(T.CenterCrop(img_size) if center_crop else T.Resize((img_size, img_size)))
    t += [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return T.Compose(t)

def l2n(x: torch.Tensor, eps=1e-8) -> torch.Tensor:
    return x / (x.norm(dim=1, keepdim=True) + eps)

# ---------- watermark overlay ----------
def _place_pos(xW, xH, wW, wH, mode):
    if mode == "center":
        return ( (xW - wW)//2, (xH - wH)//2 )
    if mode == "rb":  # right-bottom
        return ( xW - wW - 2, xH - wH - 2 )
    # random corners
    choices = [(2,2), (xW-wW-2,2), (2,xH-wH-2), (xW-wW-2,xH-wH-2)]
    return random.choice(choices)

def make_wm_pre(wm_path: str, alpha: float=0.15, scale: float=0.25, pos: str="random"):
    """返回一个对 PIL.Image 叠水印的函数"""
    if not wm_path or not os.path.isfile(wm_path):
        raise FileNotFoundError(f"watermark not found: {wm_path}")
    wm0 = Image.open(wm_path).convert("RGBA")
    alpha_i = int(max(0,min(1,alpha))*255)
    pos = pos.lower()
    def fn(img: Image.Image) -> Image.Image:
        x = img.convert("RGBA")
        xW, xH = x.size
        # 缩放
        target_w = int(xW * max(0.05, min(0.9, scale)))
        r = target_w / wm0.width
        wm = wm0.resize((target_w, max(1, int(wm0.height*r))), Image.BICUBIC)
        # 透明度
        if wm.mode != "RGBA": wm = wm.convert("RGBA")
        wmr = wm.copy()
        r,g,b,a = wmr.split()
        a = a.point(lambda v: int(v * (alpha_i/255.0)))
        wmr = Image.merge("RGBA", (r,g,b,a))
        # 位置
        xPos, yPos = _place_pos(xW, xH, wmr.width, wmr.height, "rb" if pos=="rb" else "center" if pos=="center" else "random")
        # 叠加
        canvas = x.copy()
        canvas.alpha_composite(wmr, dest=(xPos, yPos))
        return canvas.convert("RGB")
    return fn

# ---------- dataset ----------
class CsvSet(Dataset):
    def __init__(self, csv_path: str, image_root: str, transform: T.Compose, pre=None):
        super().__init__()
        self.transform = transform
        self.pre = pre
        self.samples = []
        df = pd.read_csv(csv_path)
        fields = tuple(df.columns)
        if "image_path" not in df.columns or "product_id" not in df.columns:
            raise RuntimeError(f"{csv_path} 缺少必须列: image_path, product_id；实际列={fields}")

        total, kept, miss = 0, 0, 0
        for i,r in df.iterrows():
            total += 1
            ip = str(r.get("image_path") or "").strip()
            pid = str(r.get("product_id") or "").strip()
            if not ip or not pid: miss += 1; continue
            p = ip if os.path.isabs(ip) else os.path.normpath(os.path.join(image_root, ip))
            if not os.path.isfile(p): miss += 1; continue
            self.samples.append((p, pid, i)); kept += 1

        print(f"[Data] from {csv_path}  fields={fields}")
        print(f"[Data] total_rows={total} kept={kept} missing/invalid={miss} ({(miss/max(1,total))*100:.2f}%)")
        if kept>0:
            vc = pd.Series([pid for _,pid,_ in self.samples]).value_counts().values
            print(f"[Data] images={kept} products={len(vc)} imgs/prod (min/mean/median/max)=({vc.min()}/{vc.mean():.2f}/{np.median(vc):.1f}/{vc.max()}) single-prod-ratio={(vc==1).sum()/max(1,len(vc))*100:.2f}%")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx: int):
        p, pid, raw = self.samples[idx]
        img = Image.open(p).convert("RGB")
        if self.pre is not None:
            img = self.pre(img)
        img = self.transform(img)
        return img, pid, idx

# ---------- model loading ----------
def load_model_from_ckpt(ckpt_path: str, device: torch.device,
                         model_entry: str="auto", entry_args: dict|None=None) -> nn.Module:
    """优先用项目根 quick_report.load_checkpoint；否则构建 resnet50 backbone 并映射 checkpoint 的 backbone.* 权重"""
    entry_args = entry_args or {}

    # A: local quick_report.py
    import importlib.util
    here = os.path.dirname(os.path.abspath(__file__))
    qr_path = os.path.join(here, "quick_report.py")
    if os.path.isfile(qr_path):
        spec = importlib.util.spec_from_file_location("qr_local", qr_path)
        qr_local = importlib.util.module_from_spec(spec); spec.loader.exec_module(qr_local)
        if hasattr(qr_local, "load_checkpoint"):
            try:
                model = qr_local.load_checkpoint(ckpt_path, device, model_entry, entry_args)
            except TypeError:
                model = qr_local.load_checkpoint(ckpt_path, device)
            model.eval().to(device)
            print("[LOAD] model via local quick_report.load_checkpoint")
            return model

    # B: resnet50 backbone fallback
    print("[LOAD] fallback: build ResNet50 backbone and map checkpoint weights")
    model = M.resnet50(weights=None); model.fc = nn.Identity()
    obj = torch.load(ckpt_path, map_location=device)
    state = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
    tgt = model.state_dict(); mapped = {}
    for k,v in state.items():
        nk = k[len("backbone."):] if k.startswith("backbone.") else k
        if nk in tgt and tgt[nk].shape == v.shape: mapped[nk] = v
    tgt.update(mapped)
    model.load_state_dict(tgt, strict=False)
    model.to(device).eval()
    print(f"[LOAD] resnet50 mapped keys: {len(mapped)} loaded")
    return model

# ---------- feature ----------
@torch.no_grad()
def forward_to_feat(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    f = model(x)
    if isinstance(f, (tuple,list)):
        f = next((t for t in f if hasattr(t,"dim")), f[0])
    if f.dim()==4: f = F.adaptive_avg_pool2d(f,1).flatten(1)
    elif f.dim()==1: f = f.unsqueeze(0)
    return l2n(f.float())

@torch.no_grad()
def extract_feats(model: nn.Module, loader: DataLoader, device: torch.device):
    feats, pids = [], []
    for imgs, pid, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        f = forward_to_feat(model, imgs)
        feats.append(f.cpu())
        pids += list(pid)
    return torch.cat(feats,0).float(), np.array(pids, dtype=object)

@torch.no_grad()
def recall_at_1(q_feat: torch.Tensor, q_pid: np.ndarray, g_feat: torch.Tensor, g_pid: np.ndarray) -> float:
    sim = q_feat @ g_feat.t()
    top1 = sim.argmax(dim=1).cpu().numpy()
    pred = g_pid[top1]
    return float((pred == q_pid).mean().item())

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Cross retrieval (query vs gallery)")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--query_csv", type=str, required=True)
    ap.add_argument("--gallery_csv", type=str, required=True)
    ap.add_argument("--image_root", type=str, default=".")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--center_crop", action="store_true")

    # 叠水印相关
    ap.add_argument("--query_wm", type=str, default="")
    ap.add_argument("--gallery_wm", type=str, default="")
    ap.add_argument("--wm_alpha", type=float, default=0.15)
    ap.add_argument("--wm_scale", type=float, default=0.25)
    ap.add_argument("--wm_pos", type=str, default="random", choices=["random","center","rb"])

    # gallery 双路（原图 + 叠水印）并均值
    ap.add_argument("--gallery_dual", action="store_true")

    # 可选：显式传 entry_args（一般不用）
    ap.add_argument("--model_entry", type=str, default="auto")
    ap.add_argument("--entry_args", type=str, default="{}")
    args = ap.parse_args()

    device = torch.device(args.device if (args.device=="cpu" or torch.cuda.is_available()) else "cpu")
    print(f"[Device] {device.type}")

    transform = build_transform(args.img_size, args.center_crop)

    pre_q = make_wm_pre(args.query_wm, args.wm_alpha, args.wm_scale, args.wm_pos) if args.query_wm else None
    pre_g = make_wm_pre(args.gallery_wm, args.wm_alpha, args.wm_scale, args.wm_pos) if args.gallery_wm else None

    qset = CsvSet(args.query_csv, args.image_root, transform, pre=pre_q)
    gset = CsvSet(args.gallery_csv, args.image_root, transform, pre=None if args.gallery_dual else pre_g)
    if len(qset)==0 or len(gset)==0: raise RuntimeError("query 或 gallery 为空")

    qloader = DataLoader(qset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    gloader = DataLoader(gset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 模型
    try:
        entry_args = json.loads(args.entry_args) if args.entry_args else {}
    except Exception as e:
        raise RuntimeError(f"--entry_args 需要 JSON 字符串: {e}")
    model = load_model_from_ckpt(args.ckpt, device, args.model_entry, entry_args)

    # 提特征
    print("[FEAT] extracting query features ...")
    q_feat, q_pid = extract_feats(model, qloader, device)
    print(f"[FEAT] query: feats={tuple(q_feat.shape)}")

    print("[FEAT] extracting gallery features ...")
    g_feat, g_pid = extract_feats(model, gloader, device)
    print(f"[FEAT] gallery: feats={tuple(g_feat.shape)}")

    # 可选：gallery 双路均值（原图 + 叠水印）
    if args.gallery_dual:
        print("[FEAT] extracting gallery (watermarked) features for dual-path ...")
        gset_wm = CsvSet(args.gallery_csv, args.image_root, transform, pre=pre_g)
        gloader_wm = DataLoader(gset_wm, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        g_feat_wm, _ = extract_feats(model, gloader_wm, device)
        # 归一化后均值再归一化
        g_feat = l2n(g_feat) * 0.5 + l2n(g_feat_wm) * 0.5
        g_feat = l2n(g_feat)
        print(f"[FEAT] gallery-dual fused: feats={tuple(g_feat.shape)}")

    # 计算 R@1
    r1 = recall_at_1(q_feat, q_pid, g_feat, g_pid)
    print(f"[RESULT] Cross Recall@1 (query={os.path.basename(args.query_csv)} → gallery={os.path.basename(args.gallery_csv)}) = {r1:.4f}  ({r1*100:.2f}%)")

if __name__ == "__main__":
    main()

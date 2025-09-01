#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
quick_report.py — pth权重快速评测（R@1）
用法示例：
  # 自动从ckpt推断（失败则回退到 train.MultiTaskResNet）
  python quick_report.py \
    --ckpt out/s2_e70_wminv_prod_gpu/best.pth \
    --val_csv cvs/dummy_stage_two_verification.csv \
    --image_root . \
    --device cuda \
    --batch_size 128 \
    --center_crop

  # 显式指定模型入口与参数（推荐更稳定）
  python quick_report.py \
    --ckpt out/s2_e70_wminv_prod_gpu/best.pth \
    --val_csv cvs/dummy_stage_two_verification.csv \
    --image_root . \
    --device cuda \
    --batch_size 128 \
    --center_crop \
    --model_entry MultiTaskResNet \
    --entry_args '{"embed_dim":512,"num_categories":1,"pretrained":false,"l2_norm":true}'
"""

import argparse
import csv
import io
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

# ---------------------------
# Utilities
# ---------------------------

def log(s: str) -> None:
    print(s, flush=True)

def safe_torch_load(path: str, map_location: str = "cpu") -> Any:
    """
    安全加载checkpoint；优先使用 weights_only=True（新torch支持），否则回退。
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=True)  # type: ignore[arg-type]
    except TypeError:
        return torch.load(path, map_location=map_location)

def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            out[k[len("module."):]] = v
        else:
            out[k] = v
    return out

HEAD_PREFIXES = ("category_head.", "wm_head.", "classifier.", "head.", "fc.", "inv_head.")

def is_head_key(k: str) -> bool:
    return any(k.startswith(p) for p in HEAD_PREFIXES)

def safe_load_state_dict(model: nn.Module, state: Dict[str, torch.Tensor]) -> Tuple[List[str], List[str], List[str]]:
    """
    仅加载形状匹配的参数；跳过head与不匹配者。
    返回 (loaded_keys, skipped_keys, unexpected_keys)
    """
    model_sd = model.state_dict()
    state = strip_module_prefix(state)
    loadable = {}
    skipped = []
    for k, v in state.items():
        if is_head_key(k):
            skipped.append(k)
            continue
        if (k in model_sd) and (getattr(model_sd[k], "shape", None) == getattr(v, "shape", None)):
            loadable[k] = v
        else:
            skipped.append(k)
    missing, unexpected = model.load_state_dict(loadable, strict=False)
    loaded = list(loadable.keys())
    return loaded, skipped, list(unexpected)

def try_import(entry_name: str):
    """
    尝试从常见模块导入类：优先当前目录的 train.py / model.py / models.py
    """
    # 先尝试本地模块
    for mod in ("train", "model", "models"):
        try:
            m = __import__(mod, fromlist=[entry_name])
            if hasattr(m, entry_name):
                return getattr(m, entry_name)
        except Exception:
            pass
    # 再尝试全局可见
    comps = entry_name.split(".")
    for i in range(len(comps), 0, -1):
        mod_name = ".".join(comps[:i])
        cls_name = ".".join(comps[i:])
        if not cls_name:
            continue
        try:
            m = __import__(mod_name, fromlist=[cls_name])
            if hasattr(m, cls_name):
                return getattr(m, cls_name)
        except Exception:
            continue
    raise RuntimeError(f"无法导入模型类：{entry_name}")

def build_model(model_entry: str, entry_args: Dict[str, Any]) -> nn.Module:
    """
    构建模型实例；要求模型类可用 **kwargs 初始化。
    """
    cls = try_import(model_entry)
    try:
        model = cls(**entry_args)
    except TypeError:
        # 部分模型可能不接受多余参数，做一次参数裁剪：只传它接收的参数
        import inspect
        sig = inspect.signature(cls)
        filtered = {k: v for k, v in entry_args.items() if k in sig.parameters}
        model = cls(**filtered)
    return model

def infer_from_ckpt_meta(obj: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """
    从ckpt里推断 model_entry 与 entry_args；给出合理默认值。
    """
    # 常见保存字段尝试
    model_entry = obj.get("model_entry") or obj.get("arch") or "MultiTaskResNet"
    embed_dim = int(obj.get("embed_dim", 512))
    num_categories = int(obj.get("num_categories", 1))
    backbone = obj.get("backbone", "resnet50")
    l2_norm = bool(obj.get("l2_norm", True))
    pretrained = bool(obj.get("pretrained", False))
    entry_args = {
        "embed_dim": embed_dim,
        "num_categories": num_categories,
        "backbone": backbone,
        "l2_norm": l2_norm,
        "pretrained": pretrained,
    }
    return model_entry, entry_args

# ---------------------------
# Data
# ---------------------------

class ValCsvDataset(Dataset):
    """
    只用 image_path + product_id 两列；其余列忽略。
    """
    def __init__(self, csv_path: str, image_root: str, transform: T.Compose):
        super().__init__()
        self.image_root = image_root
        self.samples: List[Tuple[str, str]] = []  # (abs_path, product_id)
        with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if "image_path" not in reader.fieldnames or "product_id" not in reader.fieldnames:
                raise RuntimeError("CSV缺少必须列：image_path / product_id")
            for row in reader:
                ip = row["image_path"]
                pid = str(row["product_id"])
                if not ip:
                    continue
                abs_p = os.path.join(image_root, ip) if not os.path.isabs(ip) else ip
                self.samples.append((abs_p, pid))
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        p, pid = self.samples[idx]
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            # 读失败给一张黑图，避免中断
            img = Image.new("RGB", (224, 224), (0, 0, 0))
        img = self.transform(img)
        return img, pid, idx

# ---------------------------
# Eval (R@1)
# ---------------------------

@torch.no_grad()
def extract_features(model: nn.Module, loader: DataLoader, device: torch.device, l2_norm: bool = True) -> Tuple[torch.Tensor, List[str]]:
    feats = []
    pids: List[str] = []
    model.eval()
    for images, pid_batch, _indices in loader:
        images = images.to(device, non_blocking=True)
        out = model(images)  # 兼容：tensor / tuple / dict
        if isinstance(out, dict):
            feat = out.get("embeddings") or out.get("embedding") or out.get("feat") or out.get("features")
            if feat is None:
                # 尝试通用键
                first_val = next(iter(out.values()))
                feat = first_val
        elif isinstance(out, (list, tuple)):
            feat = out[0]
        else:
            feat = out
        feat = feat.float()
        if l2_norm:
            feat = torch.nn.functional.normalize(feat, dim=1)
        feats.append(feat.detach().cpu())
        pids.extend(list(pid_batch))
    feats = torch.cat(feats, dim=0)
    return feats, pids

def recall_at_1(features: torch.Tensor, pids: List[str]) -> float:
    """
    leave-one-out 最近邻 (cosine)，排除自身匹配
    """
    # 归一化已在提取阶段做过；这里确保一下
    features = torch.nn.functional.normalize(features, dim=1)
    # 相似度矩阵
    sim = features @ features.t()
    n = sim.size(0)
    # 排除自身
    sim.fill_diagonal_(-1.0)
    # 每行最大相似度的索引
    nn_idx = sim.argmax(dim=1).cpu().numpy()
    correct = 0
    for i in range(n):
        if pids[i] == pids[nn_idx[i]]:
            correct += 1
    return correct / n

# ---------------------------
# Main
# ---------------------------

def build_transform(center_crop: bool) -> T.Compose:
    tfms = []
    # 经典 imagenet 预处理
    tfms.append(T.Resize(256))
    if center_crop:
        tfms.append(T.CenterCrop(224))
    else:
        tfms.append(T.RandomResizedCrop(224, scale=(0.9, 1.0)))
    tfms.append(T.ToTensor())
    tfms.append(T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]))
    return T.Compose(tfms)

def parse_entry_args(s: Optional[str]) -> Dict[str, Any]:
    if not s:
        return {}
    if isinstance(s, dict):
        return dict(s)
    s = s.strip()
    return json.loads(s)

def dataset_stats(paths: List[str], pids: List[str]) -> None:
    from collections import Counter
    cnt = Counter(pids)
    n_img = len(pids)
    n_prod = len(cnt)
    v = sorted(cnt.values())
    single = sum(1 for x in v if x == 1)
    ratio_single = single / max(1, n_prod) * 100.0
    mean_imgs = float(np.mean(v)) if v else 0.0
    median_imgs = float(np.median(v)) if v else 0.0
    log(f"[Data] images={n_img} products={n_prod} imgs/prod (min/mean/median/max)=({min(v) if v else 0}/{mean_imgs:.2f}/{median_imgs:.1f}/{max(v) if v else 0}) single-prod-ratio={ratio_single:.2f}%")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, type=str)
    p.add_argument("--val_csv", required=True, type=str)
    p.add_argument("--image_root", required=True, type=str)
    p.add_argument("--device", default="cuda", type=str, choices=["cuda","cpu","auto"])
    p.add_argument("--batch_size", default=128, type=int)
    p.add_argument("--workers", default=4, type=int)
    p.add_argument("--center_crop", action="store_true")
    p.add_argument("--model_entry", default="auto", type=str)
    p.add_argument("--entry_args", default="", type=str, help='JSON字符串，如 {"embed_dim":512,"num_categories":1,"pretrained":false,"l2_norm":true}')
    args = p.parse_args()

    # 设备
    if args.device == "auto":
        use_cuda = torch.cuda.is_available()
    else:
        use_cuda = (args.device == "cuda") and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    log(f"[Device] {device.type}")

    # 读 ckpt
    obj = safe_torch_load(args.ckpt, map_location="cpu")
    # 推断/合并模型入口与参数
    model_entry, meta_args = infer_from_ckpt_meta(obj if isinstance(obj, dict) else {})
    cli_args = parse_entry_args(args.entry_args)
    # CLI优先，ckpt次之，内置默认最后
    entry_args = dict(meta_args)
    entry_args.update(cli_args or {})

    # 构建模型
    if args.model_entry != "auto":
        model_entry = args.model_entry
    try:
        model = build_model(model_entry, entry_args)
    except Exception as e:
        # 回退到 train.MultiTaskResNet
        log(f"[WARN] auto/显式构建模型失败：{e}; 尝试 train.MultiTaskResNet 回退。")
        model_entry = "MultiTaskResNet"
        model = build_model(model_entry, entry_args)
    model.to(device).eval()

    # 加载权重
    # 1) 取 state_dict
    state = None
    if isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
        state = obj["model"]
    elif isinstance(obj, dict):
        # 兼容：直接是state_dict
        state = {k: v for k, v in obj.items() if isinstance(v, torch.Tensor)}
    else:
        raise RuntimeError("ckpt格式无法识别：既不是包含'model'的dict，也不是state_dict")
    loaded, skipped, unexpected = safe_load_state_dict(model, state)
    log(f"[LOAD] loaded={len(loaded)} skipped={len(skipped)} unexpected={len(unexpected)}")
    if skipped:
        short = ", ".join(skipped[:8]) + (f" ...(+{len(skipped)-8})" if len(skipped)>8 else "")
        log(f"[LOAD] skipped keys: {short}")

    # 数据
    transform = build_transform(center_crop=args.center_crop)
    ds = ValCsvDataset(args.val_csv, args.image_root, transform)
    if len(ds) == 0:
        raise RuntimeError("验证集为空")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=use_cuda)

    # 统计
    paths = [p for p,_pid in ds.samples]
    pids = [pid for _p, pid in ds.samples]
    dataset_stats(paths, pids)

    # 提取特征
    l2 = bool(entry_args.get("l2_norm", True))
    feats, pids = extract_features(model, loader, device, l2_norm=l2)

    # 评测
    r1 = recall_at_1(feats, pids)
    log(f"[RESULT] Recall@1 = {r1:.4f}  ({r1*100:.2f}%)")

if __name__ == "__main__":
    main()

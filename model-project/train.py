#!/usr/bin/env python3
"""
Training script for the e-commerce image retrieval model with watermark detection.

This script orchestrates data loading, model instantiation, loss definition,
training loop and evaluation. It supports multi-task learning with three
objectives: metric learning via batch-hard triplet loss, product category
classification via cross-entropy, and watermark presence detection via
binary cross-entropy with logits. Optional masking/gating is applied to
reduce the impact of watermark regions on the embedding space.

See ``README.md`` for detailed usage and description of command-line
arguments.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from datetime import datetime
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import EcommerceDataset
from sampler import PKSampler
from model import MultiTaskResNet
from losses import BatchHardTripletLoss
# from .dataset import EcommerceDataset
# from .sampler import PKSampler
# from .model import MultiTaskResNet
# from .losses import BatchHardTripletLoss
import torchvision.transforms as T

from contextlib import nullcontext
import warnings

warnings.filterwarnings("ignore", message="Initializing zero-element tensors is a no-op")


def watermark_invariance_loss(embeddings: torch.Tensor, prod_labels: torch.Tensor, wm_labels: torch.Tensor) -> torch.Tensor:
    """
    Encourage embeddings of the same product under different watermark states to be similar.

    Parameters
    ----------
    embeddings : Tensor (B,D), L2-normalized embeddings
    prod_labels : Tensor (B,), product labels
    wm_labels : Tensor (B,), watermark labels (0 for clean, 1 for watermarked)

    Returns
    -------
    Tensor: scalar loss = mean(1 - cos_sim) over pairs of the same product with different watermark labels
    """
    with torch.no_grad():
        same_prod = prod_labels.unsqueeze(1).eq(prod_labels.unsqueeze(0))
        diff_wm = wm_labels.unsqueeze(1).ne(wm_labels.unsqueeze(0))
        mask = same_prod & diff_wm
        mask = torch.triu(mask, diagonal=1)
    if mask.sum() == 0:
        return embeddings.new_tensor(0.0)
    # Cosine similarity; embeddings assumed normalized
    sim = embeddings @ embeddings.t()
    return (1.0 - sim[mask]).mean()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(model: MultiTaskResNet, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    """Evaluate the model on a validation set.

    Computes retrieval recall@1 (Top-1), category classification accuracy, and
    watermark detection accuracy.

    Returns
    -------
    dict with keys: 'top1', 'cat_acc', 'wm_acc', 'num'
    """
    model.eval()
    # 验证集为空时直接返回 0 指标
    if hasattr(loader, "dataset") and len(loader.dataset) == 0:
        return {"top1": 0.0, "cat_acc": 0.0, "wm_acc": 0.0, "num": 0}
    all_embeddings = []
    all_prod_labels = []
    all_cat_labels = []
    all_wm_labels = []
    all_cat_logits = []
    all_wm_logits = []
    with torch.no_grad():
        for imgs, prod_labels, cat_labels, wm_labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            prod_labels = prod_labels.to(device, non_blocking=True)
            cat_labels = cat_labels.to(device, non_blocking=True)
            wm_labels = wm_labels.to(device, non_blocking=True)
            emb, cat_logits, wm_logits = model(imgs)
            all_embeddings.append(emb.cpu())
            all_prod_labels.append(prod_labels.cpu())
            all_cat_labels.append(cat_labels.cpu())
            all_wm_labels.append(wm_labels.cpu())
            all_cat_logits.append(cat_logits.cpu())
            all_wm_logits.append(wm_logits.cpu())

    if not all_embeddings:
            return {"top1": 0.0, "cat_acc": 0.0, "wm_acc": 0.0, "num": 0}
    embeddings = torch.cat(all_embeddings, dim=0)
    prod_labels = torch.cat(all_prod_labels, dim=0)
    cat_labels = torch.cat(all_cat_labels, dim=0)
    wm_labels = torch.cat(all_wm_labels, dim=0).float()
    cat_logits = torch.cat(all_cat_logits, dim=0)
    wm_logits = torch.cat(all_wm_logits, dim=0)
    # Retrieval evaluation: use first instance of each product as gallery
    gallery_indices = {}
    gallery_embeddings = []
    gallery_prod_labels = []
    for idx, pid in enumerate(prod_labels.tolist()):
        if pid not in gallery_indices:
            gallery_indices[pid] = len(gallery_embeddings)
            gallery_embeddings.append(embeddings[idx])
            gallery_prod_labels.append(pid)
    gallery_embeddings = torch.stack(gallery_embeddings, dim=0)  # (G,D)
    gallery_prod_labels = torch.tensor(gallery_prod_labels)
    # Cosine similarity (embeddings are normalized)
    sims = torch.matmul(embeddings, gallery_embeddings.t())  # (N,G)
    pred_indices = sims.argmax(dim=1)
    pred_prod = gallery_prod_labels[pred_indices]
    top1 = (pred_prod == prod_labels).float().mean().item()
    # Category accuracy
    pred_cat = cat_logits.argmax(dim=1)
    cat_acc = (pred_cat == cat_labels).float().mean().item()
    # Watermark accuracy
    pred_wm = (torch.sigmoid(wm_logits) > 0.5).float()
    wm_acc = (pred_wm == wm_labels).float().mean().item()
    return {
        "top1": top1,
        "cat_acc": cat_acc,
        "wm_acc": wm_acc,
        "num": int(embeddings.size(0)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train multi-task image retrieval model with watermark detection")
    parser.add_argument("--csv_path", required=True, help="Path to training CSV file")
    parser.add_argument("--val_csv_path", default=None, help="Path to validation CSV file")
    parser.add_argument("--image_root", default="", help="Root directory for resolving image paths")
    parser.add_argument("--product_col", default="product_id", help="CSV column name for product id")
    parser.add_argument("--category_col", default="category", help="CSV column name for category")
    parser.add_argument("--image_col", default="image_path", help="CSV column name for image path")
    parser.add_argument("--mask_col", default="mask_path", help="CSV column name for mask path")
    parser.add_argument("--wm_col", default="wm_mask_path", help="CSV column name for watermarked image path")
    parser.add_argument("--mask_suffix", default="_mask.png", help="Suffix to infer mask path when mask_col missing or empty")
    parser.add_argument("--global_watermark_path", default=None, help="Path to global watermark image (PNG with alpha)")
    parser.add_argument("--alpha_threshold", type=float, default=0.5, help="Alpha threshold for global watermark")
    parser.add_argument("--no_mask_gating", action="store_true", help="Disable mask gating; do not zero out watermark regions")
    parser.add_argument("--batch_p", type=int, default=16, help="Number of classes per batch (P)")
    parser.add_argument("--batch_k", type=int, default=4, help="Number of samples per class (K)")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--embed_dim", type=int, default=512, help="Dimension of embedding vector")
    parser.add_argument("--margin", type=float, default=0.2, help="Margin for triplet loss; use None for softplus variant")
    parser.add_argument("--lambda_cat", type=float, default=0.5, help="Weight for category classification loss")
    parser.add_argument("--lambda_wm", type=float, default=1.0, help="Weight for watermark classification loss")
    parser.add_argument("--lambda_inv", type=float, default=0.3, help="Weight for watermark invariance loss")
    parser.add_argument("--prefer_s01_ratio", type=float, default=0.5, help="Fraction of classes per batch drawn from S01 cohort")
    parser.add_argument("--use_aug", action="store_true", help="Enable stronger data augmentation (RandomResizedCrop, ColorJitter, RandomErasing)")
    parser.add_argument("--use_cosine_lr", action="store_true", help="Use cosine annealing learning rate scheduler")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for optimizer")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="Device to train on")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader worker processes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--outdir", default="./outputs", help="Directory to save models and logs")
    parser.add_argument("--no_pretrain", action="store_true", help="Do not load ImageNet pretrained weights")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume training from")
    parser.add_argument("--save_each_epoch", action="store_true", help="Save model at each epoch")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    set_seed(args.seed)
    # Device selection
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Build data augmentation for training if requested
    train_transform: Optional[T.Compose] = None
    if args.use_aug:
        # Stronger augmentation: RandomResizedCrop, RandomHorizontalFlip, ColorJitter, RandomErasing
        train_transform = T.Compose([
            T.RandomResizedCrop(224, scale=(0.6, 1.0), ratio=(0.75, 1.3333), interpolation=T.InterpolationMode.BILINEAR),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            T.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0.0),
        ])
    # Dataset and DataLoader
    train_ds = EcommerceDataset(
        csv_path=args.csv_path,
        image_root=args.image_root,
        image_col=args.image_col,
        product_col=args.product_col,
        category_col=args.category_col,
        mask_col=args.mask_col,
        wm_col=args.wm_col,
        mask_suffix=args.mask_suffix,
        global_watermark_path=args.global_watermark_path,
        alpha_threshold=args.alpha_threshold,
        mask_gating=not args.no_mask_gating,
        transform=train_transform,
    )
    print(f"[SANITY] train_len={len(train_ds)}")
    # Build sampler with optional S01 preference
    prefer_labels = getattr(train_ds, "s01_label_set", None)
    sampler = PKSampler(
        train_ds.labels,
        P=args.batch_p,
        K=args.batch_k,
        shuffle=True,
        prefer_labels=prefer_labels,
        prefer_ratio=args.prefer_s01_ratio,
    )
    batch_size = args.batch_p * args.batch_k
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    # Validation dataset
    if args.val_csv_path:
        val_ds = EcommerceDataset(
            csv_path=args.val_csv_path,
            image_root=args.image_root,
            image_col=args.image_col,
            product_col=args.product_col,
            category_col=args.category_col,
            mask_col=args.mask_col,
            wm_col=args.wm_col,
            mask_suffix=args.mask_suffix,
            global_watermark_path=args.global_watermark_path,
            alpha_threshold=args.alpha_threshold,
            mask_gating=not args.no_mask_gating,
        )
        print(f"[SANITY] val_len={len(val_ds)} from {args.val_csv_path}")
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=max(1, args.num_workers // 2),
            pin_memory=(device.type == "cuda"),
        )
    else:
        val_loader = None

    # Model
    model = MultiTaskResNet(
        embed_dim=args.embed_dim,
        num_categories=train_ds.num_categories,
        pretrained=not args.no_pretrain,
        l2_norm=True,
    ).to(device)
    # Optionally resume
    start_epoch = 1
    best_top1 = 0.0
    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint.get("model", checkpoint))
            start_epoch = checkpoint.get("epoch", 1)
            best_top1 = checkpoint.get("best_top1", 0.0)
            print(f"Resumed from {args.resume}, starting at epoch {start_epoch}")
        else:
            print(f"Warning: resume checkpoint {args.resume} not found; starting from scratch")

    # Losses
    triplet_loss_fn = BatchHardTripletLoss(margin=args.margin)
    cat_loss_fn = nn.CrossEntropyLoss()
    wm_loss_fn = nn.BCEWithLogitsLoss()
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    use_cuda_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda") if use_cuda_amp else None
    # Optional cosine annealing scheduler
    scheduler = None
    if args.use_cosine_lr:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running_triplet = 0.0
        running_cat = 0.0
        running_wm = 0.0
        running_inv = 0.0
        num_batches = 0
        for imgs, prod_labels, cat_labels, wm_labels in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            prod_labels = prod_labels.to(device, non_blocking=True)
            cat_labels = cat_labels.to(device, non_blocking=True)
            wm_labels = wm_labels.float().to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            amp_ctx = torch.amp.autocast("cuda") if use_cuda_amp else nullcontext()
            with amp_ctx:
                emb, cat_logits, wm_logits = model(imgs)
                t_loss = triplet_loss_fn(emb, prod_labels)
                c_loss = cat_loss_fn(cat_logits, cat_labels)
                w_loss = wm_loss_fn(wm_logits, wm_labels)
                inv_loss = watermark_invariance_loss(emb, prod_labels, wm_labels)
                loss = (
                        t_loss
                        + args.lambda_cat * c_loss
                        + args.lambda_wm * w_loss
                        + args.lambda_inv * inv_loss
                )

            optimizer.zero_grad(set_to_none=True)
            if use_cuda_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            running_triplet += t_loss.item()
            running_cat += c_loss.item()
            running_wm += w_loss.item()
            running_inv += inv_loss.item()
            num_batches += 1
        # Compute average losses
        avg_triplet = running_triplet / max(num_batches, 1)
        avg_cat = running_cat / max(num_batches, 1)
        avg_wm = running_wm / max(num_batches, 1)
        avg_inv = running_inv / max(num_batches, 1)
        log = {
            "epoch": epoch,
            "train_triplet": avg_triplet,
            "train_cat": avg_cat,
            "train_wm": avg_wm,
            "train_inv": avg_inv,
        }
        # Evaluate
        if val_loader is not None:
            metrics = evaluate(model, val_loader, device)
            log.update(metrics)
            # Save best model
            if metrics["top1"] > best_top1:
                best_top1 = metrics["top1"]
                ckpt = {
                    "epoch": epoch + 1,
                    "model": model.state_dict(),
                    "embed_dim": args.embed_dim,
                    "num_categories": train_ds.num_categories,
                    "best_top1": best_top1,
                    "timestamp": datetime.utcnow().isoformat(),
                }
                best_path = os.path.join(args.outdir, "best.pth")
                torch.save(ckpt, best_path)
        print(json.dumps(log, ensure_ascii=False))
        # Save checkpoint each epoch if requested
        if args.save_each_epoch:
            ckpt = {
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "embed_dim": args.embed_dim,
                "num_categories": train_ds.num_categories,
                "best_top1": best_top1,
            }
            ckpt_path = os.path.join(args.outdir, f"model_epoch_{epoch}.pth")
            torch.save(ckpt, ckpt_path)

        # Step scheduler if used
        if scheduler is not None:
            scheduler.step()

    print(json.dumps({"best_top1": best_top1}, ensure_ascii=False))


if __name__ == "__main__":
    main()
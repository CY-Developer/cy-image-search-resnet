"""
train.py
~~~~~~~~

训练脚本入口。通过命令行参数指定数据集路径、训练超参数等，构建多任务网络并启动训练。

代码示例：

```bash
python train.py --csv_path data/labels.csv --image_root data/images --batch_size 32 --epochs 30
```

脚本主要步骤：

1. 解析命令行参数并配置设备（CPU/GPU）。
2. 构建 `ProductDataset` 和 `TripletDataset`。`TripletDataset` 每次迭代随机生成 anchor、positive、negative 三元组。
3. 加载 `MultiTaskModel`，设置损失函数：TripletMarginLoss 和 BCEWithLogitsLoss。
4. 进入训练循环，对每个批次同时计算嵌入损失和分类损失并反向传播。
5. 打印训练进度和损失信息。

该脚本可根据实际需求添加验证集评估、模型保存等功能。
"""

import argparse
import os
import random
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataset import ProductDataset, TripletDataset
from model import MultiTaskModel


def parse_args() -> argparse.Namespace:
    """命令行参数解析。"""
    parser = argparse.ArgumentParser(description="多任务商品图片识别训练脚本")
    parser.add_argument("--csv_path", type=str, required=True, help="标注 CSV 文件路径")
    parser.add_argument("--image_root", type=str, default="", help="图片根目录")
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--margin", type=float, default=0.2, help="Triplet Loss 的间隔")
    parser.add_argument("--embedding_dim", type=int, default=256, help="嵌入向量维度")
    parser.add_argument("--lambda_cls", type=float, default=1.0, help="分类损失权重")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="模型保存目录")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader 的 worker 数量（>0 会并行加载图像）")
    parser.add_argument("--resume", type=str, default="", help="若指定，则加载该模型继续训练（否则自动尝试读取 save_dir/model_final.pth）")
    parser.add_argument("--save_each_epoch", action='store_true', help="是否保存每个 epoch 的模型文件")
    parser.add_argument("--mask_suffix", type=str, default="_mask.png", help="自动猜测掩模文件的后缀")
    parser.add_argument("--global_watermark_path", type=str, default="", help="用于生成全局水印掩模的 PNG 或带透明通道的文件路径")
    parser.add_argument("--alpha_threshold", type=float, default=0.5, help="生成全局水印掩模时的二值化阈值 (0~1)")
    parser.add_argument("--no_mask_gating", action='store_true', help="禁用模型中的掩模 gating（默认开启）")

    # 可选验证集与评估频率
    parser.add_argument("--val_csv", type=str, default="", help="验证集 CSV（可选）")
    parser.add_argument("--val_image_root", type=str, default="", help="验证集图片根目录（可选）")
    parser.add_argument("--eval_freq", type=int, default=1, help="每多少个 epoch 做一次验证评估")

    return parser.parse_args()


def set_seed(seed: int = 42):
    """固定随机种子，保证结果可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def _select_binary_logit(logits: torch.Tensor) -> torch.Tensor:
    """
    兼容 1-logit（二分类）或 2-logits（softmax 二分类）的模型输出：
      - 若 logits 形状为 [N] 或 [N,1]，直接返回单通道；
      - 若 logits 形状为 [N,2]，取第 2 类(索引1)的 logit 作为“是水印”的正类 logit。
    返回形状：[N]
    """
    if logits.dim() == 1:
        return logits
    if logits.dim() == 2:
        if logits.size(1) == 1:
            return logits.squeeze(1)
        if logits.size(1) == 2:
            return logits[:, 1]
    # 其它形状不支持，尽量压缩到 [N]
    return logits.squeeze()


def main() -> None:
    args = parse_args()
    set_seed(42)

    # 设备选择：优先使用 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 数据预处理：缩放到 224x224，转换为张量并标准化
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 加载数据集
    base_dataset = ProductDataset(
        args.csv_path,
        image_root=args.image_root,
        transform=transform,
        mask_suffix=args.mask_suffix,
        global_watermark_path=(args.global_watermark_path if args.global_watermark_path else None),
        alpha_threshold=args.alpha_threshold,
    )

    # DataLoader（CPU 稳定：0 worker；GPU 限 4）
    n_workers = 0 if device.type == "cpu" else min(args.num_workers, os.cpu_count() or 0, 4)
    triplet_dataset = TripletDataset(base_dataset)
    dataloader = DataLoader(
        triplet_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=n_workers,
        pin_memory=False,
        persistent_workers=False,
    )

    # 创建模型（输入通道为 4：RGB + 掩模）；此处不强制你的 classifier 形状
    model = MultiTaskModel(
        embedding_dim=args.embedding_dim,
        input_channels=4,
        use_mask_gating=not args.no_mask_gating,
        # 若你的 model.__init__ 支持 pretrained 参数，可按需添加
    ).to(device)

    # 损失函数
    triplet_loss_fn = nn.TripletMarginLoss(margin=args.margin, p=2)
    bce_loss_fn = nn.BCEWithLogitsLoss()

    # 优化器 & 调度器 & AMP & 梯度裁剪
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler(enabled=(device.type == "cuda"))
    max_grad_norm = 1.0

    # 保存目录与断点续训
    os.makedirs(args.save_dir, exist_ok=True)
    # 用于跟踪训练过程中最优的 Triplet 损失和验证集 Recall@1
    best_triplet_loss = float('inf')  # 最优 Triplet 损失
    best_recall = 0.0                # 最优 Recall@1
    start_epoch = 1

    # 优先使用 --resume；否则自动尝试 save_dir/model_final.pth
    ckpt_path = args.resume if args.resume else os.path.join(args.save_dir, "model_final.pth")
    if os.path.isfile(ckpt_path):
        print(f"检测到已有模型：{ckpt_path}，尝试加载继续训练...")
        try:
            checkpoint = torch.load(ckpt_path, map_location=device)
            # 兼容两种保存方式：dict 或纯 state_dict
            model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
            if isinstance(checkpoint, dict) and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # 恢复起始 epoch（若未保存则认为从头训练）
            start_epoch = (checkpoint.get('epoch', 0) if isinstance(checkpoint, dict) else 0) + 1
            # 恢复最优指标
            if isinstance(checkpoint, dict):
                best_triplet_loss = checkpoint.get('best_triplet_loss', best_triplet_loss)
                best_recall = checkpoint.get('best_recall', best_recall)
            print(f"恢复到第 {start_epoch} 轮，best_triplet_loss={best_triplet_loss:.4f}, best_recall={best_recall:.4f}")
        except Exception as e:
            print(f"加载失败，忽略断点从头训练：{e}")

    # 日志
    log_path = os.path.join(args.save_dir, "train.log")
    log_mode = "a" if start_epoch > 1 and os.path.exists(log_path) else "w"
    log_file = open(log_path, log_mode, encoding="utf-8")
    if log_mode == "w":
        log_file.write("epoch,avg_triplet_loss,avg_cls_loss,is_best\n"); log_file.flush()

    # 训练循环
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_triplet_loss = 0.0
        epoch_cls_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")
        for images, wm_labels in progress_bar:
            # images: (anchor, positive, negative)
            anchor_imgs, pos_imgs, neg_imgs = images
            anchor_wm,  pos_wm,  neg_wm  = wm_labels  # bool 张量

            batch_size = anchor_imgs.size(0)
            imgs = torch.cat([anchor_imgs, pos_imgs, neg_imgs], dim=0).to(device)
            wm_targets = torch.cat([anchor_wm, pos_wm, neg_wm], dim=0).float().to(device)  # [N]

            optimizer.zero_grad()
            with autocast(enabled=(device.type == "cuda")):
                embeddings, raw_logits = model(imgs)

                # 拆分嵌入
                anchor_emb   = embeddings[:batch_size]
                positive_emb = embeddings[batch_size:2 * batch_size]
                negative_emb = embeddings[2 * batch_size:]

                # Triplet 损失
                loss_triplet = triplet_loss_fn(anchor_emb, positive_emb, negative_emb)

                # 二分类损失（兼容 1 或 2 logits）
                logits = _select_binary_logit(raw_logits)  # [N]
                loss_cls = bce_loss_fn(logits, wm_targets)

                # 总损失
                loss = loss_triplet + args.lambda_cls * loss_cls

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            epoch_triplet_loss += loss_triplet.item()
            epoch_cls_loss += loss_cls.item()
            num_batches += 1
            progress_bar.set_postfix({
                "triplet": f"{loss_triplet.item():.4f}",
                "cls": f"{loss_cls.item():.4f}"
            })

        scheduler.step()

        avg_triplet = epoch_triplet_loss / max(num_batches, 1)
        avg_cls = epoch_cls_loss / max(num_batches, 1)
        print(f"Epoch {epoch} 完成: Avg Triplet Loss={avg_triplet:.4f}, Avg Cls Loss={avg_cls:.4f}")

        # 保存最优模型（基于 Triplet 损失）
        is_best = avg_triplet < best_triplet_loss
        if is_best:
            best_triplet_loss = avg_triplet
            final_path = os.path.join(args.save_dir, "model_final.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_triplet_loss': best_triplet_loss,
                'best_recall': best_recall,
                'triplet_loss': avg_triplet,
                'cls_loss': avg_cls
            }, final_path)
            print(f"已更新最优模型 → {final_path} (TripletLoss={avg_triplet:.4f})")

        # 每轮快照（可选）
        if args.save_each_epoch:
            epoch_path = os.path.join(args.save_dir, f"epoch_{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_triplet_loss': best_triplet_loss,
                'best_recall': best_recall,
                'triplet_loss': avg_triplet,
                'cls_loss': avg_cls
            }, epoch_path)
            print(f"已保存模型快照: {epoch_path}")

        # 记录日志（标记是否刷新了最优 Triplet 损失）
        log_file.write(f"{epoch},{avg_triplet:.6f},{avg_cls:.6f},{1 if is_best else 0}\n"); log_file.flush()

        # 验证评估（可选）：根据 Recall@1 更新最优模型
        if args.val_csv and (epoch % max(1, args.eval_freq) == 0):
            try:
                from utils import evaluate_retrieval  # 延迟导入
                # 与训练保持一致的预处理
                val_tfm = transform
                # 构建查询集和库集（这里使用同一验证集）
                query_ds = ProductDataset(args.val_csv, image_root=args.val_image_root,
                                           transform=val_tfm, mask_suffix=args.mask_suffix)
                gallery_ds = ProductDataset(args.val_csv, image_root=args.val_image_root,
                                            transform=val_tfm, mask_suffix=args.mask_suffix)
                r1, r5, map_ = evaluate_retrieval(model, query_ds, gallery_ds, device=str(device))
                print(f"[Val] Recall@1={r1:.4f} Recall@5={r5:.4f} mAP={map_:.4f}")
                # 更新基于 Recall@1 的最佳模型
                if r1 > best_recall:
                    best_recall = r1
                    ckpt_path = os.path.join(args.save_dir, "model_final.pth")
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_triplet_loss': best_triplet_loss,
                        'best_recall': best_recall
                    }, ckpt_path)
                    print(f"更新并保存最佳模型 {ckpt_path} (Recall@1={r1:.4f})")
            except Exception as e:
                print(f"[Val] 跳过评估（未找到 evaluate_retrieval 或出错）：{e}")

    print("训练完成！")
    log_file.close()


if __name__ == "__main__":
    main()

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
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
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
    parser.add_argument("--num_workers", type=int, default=4,help="DataLoader 的 worker 数量（>0 会并行加载图像）")
    parser.add_argument("--resume", type=str, default="", help="若指定，则加载已有模型继续训练")
    parser.add_argument("--save_each_epoch", action='store_true', help="是否保存每个 epoch 的模型文件")
    parser.add_argument("--mask_suffix", type=str, default="_mask.png", help="自动猜测掩模文件的后缀")
    parser.add_argument("--global_watermark_path", type=str, default="", help="用于生成全局水印掩模的 PNG 或带透明通道的文件路径")
    parser.add_argument("--alpha_threshold", type=float, default=0.5, help="生成全局水印掩模时的二值化阈值 (0~1)")
    parser.add_argument("--no_mask_gating", action='store_true', help="禁用模型中的掩模 gating（默认开启）")
    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()
    # 设备选择：优先使用 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 数据预处理：缩放到 224x224，转换为张量并标准化
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载数据集
    base_dataset = ProductDataset(
        args.csv_path,
        image_root=args.image_root,
        transform=transform,
        mask_suffix=args.mask_suffix,
        global_watermark_path=(args.global_watermark_path if args.global_watermark_path else None),
        alpha_threshold=args.alpha_threshold
    )
    # 为了避免 “bus error / shared-memory” 报错：
    #   • CPU 训练直接设 num_workers = 0        （单线程最稳）
    #   • GPU 训练最多用 4 个线程，再根据机器 CPU 核心数裁剪
    # 同时关闭 pin_memory、persistent_workers，节省内存映射
    n_workers = 0 if device.type == "cpu" else min(args.num_workers, os.cpu_count(), 4)

    triplet_dataset = TripletDataset(base_dataset)
    dataloader = DataLoader(
        triplet_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=n_workers,
        pin_memory=False,
        persistent_workers=False
    )


    # 创建模型（输入通道为 4：RGB + 掩模）
    model = MultiTaskModel(
        embedding_dim=args.embedding_dim,
        input_channels=4,
        use_mask_gating=not args.no_mask_gating
    )
    model = model.to(device)

    # 定义损失函数
    triplet_loss_fn = nn.TripletMarginLoss(margin=args.margin, p=2)
    bce_loss_fn = nn.BCEWithLogitsLoss()

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    # 加载已有模型继续训练。初始化最佳 Triplet Loss 与起始 epoch
    best_triplet_loss = float('inf')
    start_epoch = 1
    if args.resume:
        resume_path = args.resume
        if os.path.isfile(resume_path):
            print(f"加载已有模型：{resume_path}")
            checkpoint = torch.load(resume_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint.get('epoch', 0) + 1
            best_triplet_loss = checkpoint.get('best_triplet_loss', float('inf'))
            print(f"恢复训练，从第 {start_epoch} 轮开始，最佳 Triplet Loss 为 {best_triplet_loss:.4f}")
        else:
            print(f"警告：未找到恢复模型 {resume_path}，从头开始训练。")
    # 日志文件用于记录每个 epoch 的损失指标
    log_path = os.path.join(args.save_dir, "train.log")
    # 当从头开始训练时，重写日志文件并写入表头；否则追加
    if start_epoch == 1:
        log_file = open(log_path, "w", encoding="utf-8")
        log_file.write("epoch,avg_triplet_loss,avg_cls_loss,is_best\n")
        log_file.flush()
    else:
        log_file = open(log_path, "a", encoding="utf-8")

    # 训练循环
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_triplet_loss = 0.0
        epoch_cls_loss = 0.0
        num_batches = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch_idx, (images, wm_labels) in enumerate(progress_bar):
            # images: tuple of (anchor, positive, negative)
            anchor_imgs, pos_imgs, neg_imgs = images
            anchor_wm, pos_wm, neg_wm = wm_labels
            batch_size = anchor_imgs.size(0)
            # 拼接三种图片，以便一次前向
            imgs = torch.cat([anchor_imgs, pos_imgs, neg_imgs], dim=0).to(device)
            # 合并水印标签
            wm_targets = torch.cat([
                anchor_wm.float(), pos_wm.float(), neg_wm.float()
            ], dim=0).to(device)

            optimizer.zero_grad()
            embeddings, logits = model(imgs)
            # 分割嵌入向量
            anchor_emb = embeddings[:batch_size]
            positive_emb = embeddings[batch_size:2 * batch_size]
            negative_emb = embeddings[2 * batch_size:]
            # Triplet 损失
            loss_triplet = triplet_loss_fn(anchor_emb, positive_emb, negative_emb)
            # 二分类损失：BCEWithLogitsLoss
            loss_cls = bce_loss_fn(logits, wm_targets)
            # 总损失 = Triplet + lambda * 分类
            loss = loss_triplet + args.lambda_cls * loss_cls
            loss.backward()
            optimizer.step()

            epoch_triplet_loss += loss_triplet.item()
            epoch_cls_loss += loss_cls.item()
            num_batches += 1
            progress_bar.set_postfix({
                "triplet_loss": f"{loss_triplet.item():.4f}",
                "cls_loss": f"{loss_cls.item():.4f}"
            })

        avg_triplet = epoch_triplet_loss / max(num_batches, 1)
        avg_cls = epoch_cls_loss / max(num_batches, 1)
        print(f"Epoch {epoch} 完成: Avg Triplet Loss={avg_triplet:.4f}, Avg Cls Loss={avg_cls:.4f}")

        # 判断是否为最佳模型，若是则更新并保存为 model_final.pth
        is_best = avg_triplet < best_triplet_loss
        if is_best:
            best_triplet_loss = avg_triplet
            final_path = os.path.join(args.save_dir, "model_final.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_triplet_loss': best_triplet_loss,
                'triplet_loss': avg_triplet,
                'cls_loss': avg_cls
            }, final_path)
            print(f"已更新最优模型 → {final_path} (TripletLoss={avg_triplet:.4f})")
        # 若启用保存每个 epoch
        if args.save_each_epoch:
            epoch_path = os.path.join(args.save_dir, f"epoch_{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_triplet_loss': best_triplet_loss,
                'triplet_loss': avg_triplet,
                'cls_loss': avg_cls
            }, epoch_path)
            print(f"已保存模型快照: {epoch_path}")

        # 将当前 epoch 的损失写入日志文件
        if log_file:
            log_file.write(f"{epoch},{avg_triplet:.6f},{avg_cls:.6f},{1 if is_best else 0}\n")
            log_file.flush()

    print("训练完成！")
    # 关闭日志文件
    if log_file:
        log_file.close()
def load_existing_model(model, checkpoint_path):
    if os.path.exists(checkpoint_path):
        print(f"找到现有模型：{checkpoint_path}，正在加载...")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint['epoch'], checkpoint['loss']  # 你可以根据需求加载其他信息
    else:
        print("没有找到现有模型，开始从头训练...")
        return 0, 0  # 若没有现有模型，返回从0开始


if __name__ == "__main__":
    main()
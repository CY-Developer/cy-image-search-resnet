
"""
utils.py
~~~~~~~~

提供一些辅助函数，例如计算图片嵌入向量、构建向量索引以及查询相似商品。

当前仅实现简单的特征提取函数，可在模型训练完成后使用。未来可以结合 FAISS 等库建立高效检索索引。
"""

from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import ProductDataset
from model import MultiTaskModel


@torch.no_grad()
def extract_embeddings(model: MultiTaskModel, dataset: ProductDataset, batch_size: int = 64, device: str = "cpu") -> Tuple[List[str], torch.Tensor]:
    """提取数据集中所有图片的嵌入向量。

    参数：
        model: 训练好的多任务模型。
        dataset: 包含所有图片的 ProductDataset。
        batch_size: 批处理大小。
        device: 使用的设备（"cpu" 或 "cuda"）。

    返回：
        product_ids: 按顺序排列的商品编号列表。
        embeddings: 对应图片的嵌入张量，形状为 [N, embedding_dim]。
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_embeddings: List[torch.Tensor] = []
    all_product_ids: List[str] = []
    for images, product_ids, _, _ in loader:
        images = images.to(device)
        emb, _ = model(images)
        all_embeddings.append(emb.cpu())
        all_product_ids.extend(product_ids)
    embeddings = torch.cat(all_embeddings, dim=0)
    return all_product_ids, embeddings


@torch.no_grad()
def evaluate_retrieval(model: MultiTaskModel, query_dataset: ProductDataset, gallery_dataset: ProductDataset,
                       device: str = "cpu", top_ks=(1, 5, 10)):
    """
    简易检索评估：计算 Recall@K 和 mAP（基于 Cosine 相似度）。
    这里假定同 product_id 为正样本。
    """
    model.eval()
    q_ids, q_emb = extract_embeddings(model, query_dataset, batch_size=64, device=device)
    g_ids, g_emb = extract_embeddings(model, gallery_dataset, batch_size=64, device=device)

    q = q_emb.numpy()
    g = g_emb.numpy()
    q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
    g = g / (np.linalg.norm(g, axis=1, keepdims=True) + 1e-12)

    sims = np.matmul(q, g.T)  # [Q, G]

    correct_at_k = {k: 0 for k in top_ks}
    APs = []

    for i, q_pid in enumerate(q_ids):
        order = np.argsort(-sims[i])  # 大到小
        ranked_pids = [g_ids[j] for j in order]

        # Recall@K
        for k in top_ks:
            if q_pid in ranked_pids[:k]:
                correct_at_k[k] += 1

        # AP/mAP
        hits = 0
        precisions = []
        for rank, pid in enumerate(ranked_pids, start=1):
            if pid == q_pid:
                hits += 1
                precisions.append(hits / rank)
        APs.append(np.mean(precisions) if precisions else 0.0)

    recall = {k: correct_at_k[k] / max(1, len(q_ids)) for k in top_ks}
    mAP = float(np.mean(APs)) if APs else 0.0
    # 返回常用指标：R@1、R@5、mAP
    return recall.get(1, 0.0), recall.get(5, 0.0), mAP
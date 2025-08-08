"""
utils.py
~~~~~~~~

提供一些辅助函数，例如计算图片嵌入向量、构建向量索引以及查询相似商品。

当前仅实现简单的特征提取函数，可在模型训练完成后使用。未来可以结合 FAISS 等库建立高效检索索引。
"""

from typing import List, Tuple

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

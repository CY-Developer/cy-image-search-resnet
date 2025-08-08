"""
model_service.py
~~~~~~~~~~~~~~~~~

本模块封装了一个图像嵌入服务，用于加载训练好的模型、预处理图片、生成嵌入向量，并与 Milvus、Redis 等外部组件交互。它提供了一组方法，可以在 REST 或 gRPC 服务中调用。
"""

from typing import List, Optional, Tuple, Dict

import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import redis
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

from .config import Config
from .model import load_model
from .preprocess import Preprocessor

import io


class ImageEmbeddingService:
    """商品图片向量化服务。"""

    def __init__(self, config: Config = Config()) -> None:
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # 加载模型
        self.model = load_model(config.MODEL_PATH, device=self.device,
                               embedding_dim=config.MILVUS_DIMENSION,
                               use_mask_gating=config.USE_MASK_GATING)
        # 初始化预处理器
        self.preprocessor = Preprocessor(device=self.device)
        # 图像转换：调整大小、转 Tensor、标准化
        self.transform = transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # 生成全局水印掩模
        self.global_mask: Optional[torch.Tensor] = None
        if config.GLOBAL_WATERMARK_PATH:
            try:
                with Image.open(config.GLOBAL_WATERMARK_PATH) as wm:
                    # 使用 alpha 通道或灰度通道
                    if wm.mode in ("RGBA", "LA"):
                        alpha = wm.split()[-1]
                        m = alpha
                    else:
                        m = wm.convert("L")
                    # 与 transform 保持一致的缩放
                    m = m.resize((config.IMAGE_SIZE, config.IMAGE_SIZE))
                    m_tensor = transforms.ToTensor()(m)
                    self.global_mask = (m_tensor > config.ALPHA_THRESHOLD).float()
            except Exception as e:
                print(f"无法加载全局水印掩模: {e}")
                self.global_mask = None
        # 初始化 Milvus 连接
        connections.connect("default", host=config.MILVUS_HOST, port=config.MILVUS_PORT)
        # 初始化 Redis 连接（保留业务相关配置，不做改动）
        self.redis_client = redis.Redis(host=config.REDIS_HOST, port=config.REDIS_PORT, db=config.REDIS_DB, password=config.REDIS_PASSWORD, decode_responses=False)
        # 如果需要，在初始化时创建 Milvus collection
        self._init_milvus_collection()

    def _init_milvus_collection(self) -> None:
        """创建 Milvus collection（如果不存在）。"""
        name = self.config.MILVUS_COLLECTION
        dim = self.config.MILVUS_DIMENSION
        if not utility.has_collection(name):
            print(f"Milvus 集合 {name} 不存在，正在创建……")
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim)
            ]
            schema = CollectionSchema(fields, description="product embeddings")
            collection = Collection(name, schema)
            collection.create_index(field_name="embedding", index_params={"index_type": "IVF_FLAT", "metric_type": self.config.MILVUS_METRIC_TYPE, "params": {"nlist": 128}})
            print("Milvus 集合创建完成。")
        else:
            print(f"Milvus 集合 {name} 已存在。")

    def _prepare_image(self, image: Image.Image, category: str = "") -> torch.Tensor:
        """对图片执行预处理，包括人物抑制、裁剪和尺寸标准化，生成 4 通道输入。"""
        # 去除人物干扰
        processed = self.preprocessor.remove_person(image)
        # 根据商品类别进行裁剪
        processed = self.preprocessor.crop_by_category(processed, category)
        # 调整大小并标准化
        img_tensor = self.transform(processed)
        # 构造掩模：默认全零
        mask = torch.zeros((1, self.config.IMAGE_SIZE, self.config.IMAGE_SIZE), dtype=torch.float32)
        # 使用全局掩模（若存在）
        if self.global_mask is not None:
            mask = torch.max(mask, self.global_mask)
        # 拼接为 [4,H,W]
        img_with_mask = torch.cat([img_tensor, mask], dim=0)
        return img_with_mask

    def embed_image(self, image: Image.Image, category: str = "") -> np.ndarray:
        """生成单张图片的嵌入向量。

        Args:
            image: PIL 图像实例。
            category: 商品类别，用于裁剪策略。
        Returns:
            嵌入向量，形状为 [embedding_dim]。
        """
        img_tensor = self._prepare_image(image, category)
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        with torch.no_grad():
            embeddings, _ = self.model(img_tensor)
        return embeddings.squeeze(0).cpu().numpy()

    def embed_image_bytes(self, image_bytes: bytes, category: str = "") -> np.ndarray:
        """从二进制数据生成嵌入向量。"""
        with Image.open(io.BytesIO(image_bytes)) as img:
            img = img.convert("RGB")
            return self.embed_image(img, category)

    def insert_embedding(self, embedding: np.ndarray) -> int:
        """将向量插入 Milvus 并返回生成的 ID。"""
        collection = Collection(self.config.MILVUS_COLLECTION)
        entities = [embedding.tolist()]
        ids = collection.insert([entities])[0]
        collection.flush()
        return ids[0]

    def search_embedding(self, embedding: np.ndarray, top_k: int = 10) -> List[Tuple[int, float]]:
        """在 Milvus 中搜索相似向量，返回 (id, score) 列表。"""
        collection = Collection(self.config.MILVUS_COLLECTION)
        search_params = {"metric_type": self.config.MILVUS_METRIC_TYPE, "params": {"nprobe": 10}}
        results = collection.search(data=[embedding.tolist()], anns_field="embedding", param=search_params, limit=top_k)
        matches: List[Tuple[int, float]] = []
        for hit in results[0]:
            matches.append((hit.id, hit.distance))
        return matches

    # 以下示例保留业务 API key & Redis JSON 不变，但可在此扩展缓存逻辑
    def cache_embedding(self, key: str, embedding: np.ndarray) -> None:
        """示例缓存：将嵌入向量存入 Redis。业务中请保持 API key 和 JSON 结构不变。"""
        # 将向量序列化为二进制或 JSON
        value = np.array(embedding, dtype=np.float32).tobytes()
        # 业务系统可能需要使用特定 key，例如 "product:<id>" 等
        self.redis_client.set(key, value)

"""
config.py
~~~~~~~~~~

本模块定义服务的配置类 `Config`。在部署或开发过程中，可以通过环境变量或修改此文件来
调整模型路径、Redis 连接、API 密钥等参数。

请勿随意修改业务相关的 apiKey 字段名称或 Redis 数据结构，以保证与现有 Java
服务兼容。
"""

from typing import Optional, List


class Config:
    """配置项集合。

    - MODEL_PATH: 训练好的模型权重路径。
    - REDIS_*: Redis 连接参数，用于缓存向量。
    - API_KEYS: 允许访问服务的 API 密钥列表。
    - IMAGE_SIZE: 图像缩放尺寸（正方形）。
    - USE_MASK_GATING: 是否启用水印掩模 gating。
    - GLOBAL_WATERMARK_PATH: 全局水印 PNG 文件路径，可生成统一掩模。
    - ALPHA_THRESHOLD: 生成掩模时的 alpha 通道阈值。
    - CROPPING_ENABLED: 是否根据品类裁剪图像。
    """

    # 模型权重文件路径
    MODEL_PATH: str = "model_final.pth"

    # Redis 配置
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = None

    # API 密钥列表，若列表为空则不进行校验
    API_KEYS: List[str] = ["your-secret-key"]

    # 图片预处理尺寸
    IMAGE_SIZE: int = 224

    # 模型参数
    USE_MASK_GATING: bool = True
    GLOBAL_WATERMARK_PATH: Optional[str] = None
    ALPHA_THRESHOLD: float = 0.5

    # 是否启用按品类裁剪
    CROPPING_ENABLED: bool = True
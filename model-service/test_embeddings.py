"""
test_embeddings.py
~~~~~~~~~~~~~~~~~~~

该脚本用于验证训练好的模型能否正确生成商品图片的嵌入向量。
使用示例：

    python test_embeddings.py --model_path /path/to/model_final.pth --images img1.jpg img2.jpg img3.jpg

脚本会依次读取提供的图片，进行预处理并生成向量，然后打印向量内容。您可以将这些向量与 Milvus 中的向量或其他模型的输出作对比，以判断模型的有效性。

"""

import argparse
import os
from typing import List

import torch
from PIL import Image

import torchvision  # 新增导入，以便在主函数中使用 torchvision.transforms

from model import load_model
from preprocess import Preprocessor
from config import Config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="测试嵌入向量生成")
    parser.add_argument("--model_path", type=str, required=True, help="训练模型权重路径")
    parser.add_argument("--images", type=str, nargs='+', required=True, help="待测试的图片路径列表")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = Config()
    config.MODEL_PATH = args.model_path
    # 加载模型与预处理器
    model = load_model(args.model_path, device=device, embedding_dim=config.MILVUS_DIMENSION, use_mask_gating=config.USE_MASK_GATING)
    preprocessor = Preprocessor(device=device)
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    global_mask = None
    if config.GLOBAL_WATERMARK_PATH:
        try:
            with Image.open(config.GLOBAL_WATERMARK_PATH) as wm:
                if wm.mode in ("RGBA", "LA"):
                    alpha = wm.split()[-1]
                    m = alpha
                else:
                    m = wm.convert("L")
                m = m.resize((config.IMAGE_SIZE, config.IMAGE_SIZE))
                m_tensor = torchvision.transforms.ToTensor()(m)
                global_mask = (m_tensor > config.ALPHA_THRESHOLD).float()
        except Exception:
            global_mask = None
    for img_path in args.images:
        if not os.path.exists(img_path):
            print(f"文件不存在: {img_path}")
            continue
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            # 简单预处理：去人像并中心裁剪
            processed = preprocessor.remove_person(img)
            processed = preprocessor.crop_center(processed, ratio=0.8)
            img_tensor = transform(processed)
            # 构造掩模
            mask = torch.zeros((1, config.IMAGE_SIZE, config.IMAGE_SIZE))
            if global_mask is not None:
                mask = torch.max(mask, global_mask)
            x = torch.cat([img_tensor, mask], dim=0).unsqueeze(0).to(device)
            with torch.no_grad():
                emb, _ = model(x)
            emb_np = emb.squeeze(0).cpu().numpy()
            print(f"{img_path} 的嵌入向量: {emb_np.tolist()}")


if __name__ == "__main__":
    main()
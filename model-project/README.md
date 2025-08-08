# 电商商品图片识别模型项目

本项目旨在提供一个企业级电商图像识别模型的训练方案，用于鞋子、包包、手表、珠宝等商品类别的检索与识别。项目采用多任务学习框架：同时学习商品的向量化表示（用于同款检索）和水印分类。整体结构清晰，代码可直接运行。以下内容介绍了目录结构、使用方法以及核心思想。

## 目录结构

```
ecommerce_image_retrieval/
├── README.md            # 项目说明
├── requirements.txt     # Python 环境依赖列表
├── dataset.py           # 数据集加载与样本采样逻辑
├── model.py             # 多任务神经网络模型定义
├── train.py             # 训练脚本
└── utils.py             # 辅助工具函数
```

## 环境准备

建议使用 Python 3.8 以上版本，并安装 PyTorch 1.12 以上版本（CUDA 或 CPU 均可）。安装依赖可以通过以下命令完成：

```bash
pip install -r requirements.txt
```

## 数据准备

1. **图片收集**：将商城中每件商品的主图和附图整理到磁盘，不建议初期使用复杂的详情图。主图背景干净，附图可能含有水印。
2. **标注清单**：准备一个 CSV 文件，每一行代表一张图片，至少包含四列：

   ```
   image_path,product_id,category,is_watermark[,mask_path]
   ```

   - `image_path`：图片的相对或绝对路径。
   - `product_id`：商品编号，同一商品的图片使用相同的编号。
   - `category`：商品所属类别（如鞋子、包包等）。
   - `is_watermark`：是否存在水印，`true` 或 `false`。
   - `mask_path`（可选）：水印掩模文件路径，用于描述水印覆盖区域。掩模应为灰度图，白色（像素值为 255）表示水印区域，黑色（0）表示非水印区域。若未提供，则默认根据 `mask_suffix` 自动推断（示例中为在图片名后加 `_mask.png`）。

   **全局水印掩模（可选）**：当所有图片使用同一水印（例如公司 logo）且其位置固定时，可以将该水印源文件（如 PSD 导出的 PNG）通过命令行参数 `--global_watermark_path` 传入训练脚本。数据加载器会自动读取该文件的透明通道或灰度值生成一个统一的掩模，并与单张图片的掩模合并。这使得模型在训练时能够更精准地识别和抑制水印区域，提高后续检索的鲁棒性。可以通过 `--alpha_threshold` 调整二值化阈值（范围 0~1），以控制水印区域的范围。

3. **数据增强**：训练过程中可在 `train.py` 中启用随机裁剪、颜色抖动等增强策略，以扩充样本多样性。

## 训练模型

运行以下命令开始训练（仅示例，实际参数可根据需要调整）：

```bash
# 基础训练命令
python train.py \
  --csv_path path/to/your_data.csv \
  --image_root path/to/images \
  --batch_size 32 \
  --epochs 20

# 如有统一水印源文件，可追加以下参数以启用全局水印掩模：
# --global_watermark_path path/to/watermark.png --alpha_threshold 0.5

# 若希望禁用掩模 gating（不对水印区域进行零化处理），可追加：
# --no_mask_gating
```

主要参数说明：

| 参数名         | 说明                                      |
|---------------|-------------------------------------------|
| `csv_path`    | 指向数据标注 CSV 文件的路径。              |
| `image_root`  | 图片根目录，`dataset.py` 会自动拼接路径。    |
| `batch_size`  | 批次大小，需根据显存大小进行调整。          |
| `epochs`      | 训练轮数。                                  |
| `lr`          | 学习率，默认为 `1e-4`。                     |
| `margin`      | Triplet 损失的间隔（margin），默认 `0.2`。   |
| `lambda_cls`  | 分类损失权重，控制 Triplet 和分类的平衡。    |
| `resume`      | 指定已有模型路径，从该 checkpoint 继续训练。 |
| `save_each_epoch` | 若设为 true，则每个 epoch 保存一次快照。 |
| `mask_suffix` | 自动猜测水印掩模文件的后缀，默认 `_mask.png`。 |
| `global_watermark_path` | 指定一个带透明通道的水印源文件（PNG），用于生成全局水印掩模。 |
| `alpha_threshold` | 提取水印掩模时的二值化阈值（0~1），默认为 0.5。 |
| `no_mask_gating` | 禁用掩模 gating；启用时 (默认) 会将掩模区域的像素置零，进一步降低水印干扰。 |

训练脚本会按多任务方式同时优化两种目标：

1. **商品特征嵌入**：使用 Triplet Margin Loss 使同一商品的图片在特征空间中更接近，不同商品更远。Triplet 损失的目标是确保 `dist(anchor, positive) + margin < dist(anchor, negative)`【123160963696029†L163-L174】。
2. **水印识别**：使用二分类交叉熵（`BCEWithLogitsLoss`）预测图片是否包含水印。

## 模型设计

模型采用预训练的 ResNet50 作为共享骨干，通过两个独立分支实现多任务：

1. **嵌入分支**：从共享特征中提取 256 维的向量，并经 L2 归一化用于同款检索。Triplet Loss 迫使同类向量靠近、异类向量远离【262319355435192†L64-L120】。
2. **水印分类分支**：对共享特征进行线性映射，输出一个标量 logits，通过 sigmoid 得到水印概率。采用二分类交叉熵进行训练。

## 采样策略

由于 Triplet Loss 需要 anchor、positive、negative 三元组，本项目提供 `TripletDataset` 用于在每次 `__getitem__` 时随机采样正样本和负样本。具体策略如下：

1. **Anchor**：从整个数据集中随机选择一个样本。
2. **Positive**：在同一个商品编号中随机选择另一个不同的样本。
3. **Negative**：从不同商品编号中随机选择一个样本。

该策略会在训练阶段持续随机生成三元组，充分利用数据集中的多角度图片。

## 后续工作

* 在训练完成后，可使用嵌入分支输出的向量作为商品表征，构建向量库，实现相似商品检索。
* 复杂的详情图或遮挡场景可在模型初步收敛后逐步加入训练，以增强模型鲁棒性。
* 可尝试替换 Triplet Loss 为 ArcFace 等更强大的对比损失，以进一步提升性能。

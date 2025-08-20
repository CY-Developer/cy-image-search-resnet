# 电商商品图片识别模型项目（优化版）

本项目提供一个面向企业级应用的电商图像识别模型训练框架，支持鞋子、包包、手表、珠宝等商品的同款检索，并在训练过程中同时学习**水印检测**任务。相比初始版本，本项目引入了水印分类分支和可选的水印掩模 gating，提高了模型在有水印场景下的鲁棒性和精度。

## 目录结构

```
ecommerce_image_retrieval/
├── README.md           # 项目说明
├── requirements.txt    # Python 环境依赖
├── dataset.py          # 数据集加载与掩模处理
├── model.py            # 多任务神经网络模型定义
├── train.py            # 训练脚本
└── utils.py            # 掩模处理等辅助函数
```

## 环境准备

项目基于 Python 3.8 及以上版本，核心依赖包含：

* `torch>=1.13`
* `torchvision>=0.14`
* `Pillow`、`numpy`

可通过以下命令安装所需依赖（需要具备可用的 PyPI 或预下载的 wheel 包）：

```bash
pip install -r requirements.txt
```

## 数据格式

训练数据使用 CSV 文件描述，至少包含以下列（字段名可通过命令行参数指定）：

| 列名           | 含义                                                                   |
|----------------|----------------------------------------------------------------------|
| `image_path`   | 图片路径，相对 `--image_root` 或绝对路径。                              |
| `product_id`   | 商品编号，同一商品的所有图片使用相同的编号。                            |
| `category`     | 商品所属类目（如 shoes、bags、watch 等）。                               |
| `mask_path`    | *可选*，指向该图片对应的水印掩模文件（灰度图，白色表示水印区域）。缺失或为空时认为无水印。 |

> 与最初版本不同，水印信息不再使用布尔值表示，而是直接提供掩模文件。掩模文件应与图片尺寸一致或可通过 `mask_suffix` 自动推断。例如，若图片名为 `abc.jpg`，掩模可能命名为 `abc_mask.png`。

**全局水印**：当所有图片的水印来源相同且位置固定时，可通过 `--global_watermark_path` 指定一个带透明通道的 PNG 文件（或 PSD 转换后的 PNG），训练脚本会根据其透明通道生成一个全局掩模。这可以显著提升水印检测精度。阈值由 `--alpha_threshold` 控制。

## 训练脚本

`train.py` 提供了灵活的命令行接口，支持多任务训练、掩模 gating、超参调整和模型保存。常用命令示例如下：

```bash
python train.py \ 
  --csv_path path/to/train.csv \                # 训练集 CSV
  --val_csv_path path/to/val.csv \              # 验证集 CSV（可选）
  --image_root path/to/images \                 # 图片根目录
  --epochs 20 \                                 # 训练轮数
  --batch_p 16 --batch_k 4 \                    # 每批次采样的类别数和每类样本数
  --embed_dim 512 \                             # 嵌入向量维度
  --lambda_cat 0.5 --lambda_wm 1.0 \            # 分类损失和水印损失权重
  --global_watermark_path path/to/wm.png \       # 全局水印文件（可选）
  --alpha_threshold 0.5 \                       # 全局水印透明度阈值
  --no_mask_gating \                            # 禁用掩模 gating（调试用）
  --mask_suffix _mask.png \                     # 自动推断掩模文件的后缀
  --device auto \                               # 训练设备：auto/cuda/cpu
  --outdir outputs                              # 模型输出目录
```

主要参数说明：

| 参数               | 说明                                                                                                                |
|--------------------|-------------------------------------------------------------------------------------------------------------------|
| `csv_path`         | 训练集 CSV 文件路径。                                                                                              |
| `val_csv_path`     | 验证集 CSV 文件路径（可选）。                                                                                    |
| `image_root`       | 图片根目录，数据集加载器会将 `image_path` 拼接在其后。                                                             |
| `batch_p` / `batch_k` | P-K 采样策略中类别数与每类采样个数，batch_size = P × K。                                                         |
| `epochs`           | 训练轮数。                                                                                                        |
| `margin`           | Triplet 损失的 margin，默认 0.2。                                                                                |
| `embed_dim`        | 嵌入向量维度。                                                                                                    |
| `lambda_cat`       | 类目分类损失权重，控制 Triplet 与类别损失的平衡。                                                                  |
| `lambda_wm`        | 水印分类损失权重。                                                                                                  |
| `global_watermark_path` | 全局水印文件（PNG），透明通道用于生成统一的水印掩模。                                                         |
| `alpha_threshold`  | 提取全局水印掩模时的二值化阈值（0~1）。                                                                            |
| `mask_suffix`      | 自动推断局部水印掩模时的后缀，例如 `_mask.png`。                                                                    |
| `no_mask_gating`   | 禁用掩模 gating：启用 gating 时会将水印区域像素清零，减少噪声；禁用时保持原图。                                      |
| `resume`           | 从已有模型路径恢复训练（可选）。                                                                                    |
| `save_each_epoch`  | 若设为 true，则每个 epoch 保存一次模型快照。                                                                       |
| `device`           | 训练设备，可选 auto（自动选择 GPU/CPU）、cuda 或 cpu。                                                            |
| `no_pretrain`      | 不加载 ImageNet 预训练权重。                                                                                        |

## 模型设计

模型基于 ResNet50 骨干，并通过三条分支实现多任务学习：

1. **嵌入分支**：将骨干特征投影到一个低维向量空间，并进行 L2 归一化，用于同款检索。利用 batch-hard Triplet Loss 让同一商品的向量更接近，不同商品更远。
2. **类目分类分支**：输出商品类目的 logits，使用交叉熵损失训练。在商品类目信息不足或不准确时，可通过调整 `lambda_cat` 减小其影响。
3. **水印分类分支**：输出水印存在与否的 logits，使用二分类交叉熵 (`BCEWithLogitsLoss`) 训练。配合掩模 gating，可提升在有水印图片上的检索效果。

## 掩模处理

水印掩模的作用有两点：一是作为监督信号，用于训练模型判断图片是否有水印；二是用于 **gating**，即在输入图像上将水印区域像素置零，以减弱水印带来的噪声。

- **局部掩模**：若 CSV 中提供了 `mask_path` 列，则数据加载器会读取该文件并作为局部掩模。掩模应为灰度图（单通道），白色（255）表示水印区域，黑色（0）表示非水印区域。加载后会缩放并中心裁剪，使其与模型输入尺寸对齐。
- **全局掩模**：通过 `--global_watermark_path` 指定带透明通道的水印文件，脚本会从其 alpha 通道或灰度信息中提取水印区域。若局部掩模不存在，则使用全局掩模。局部和全局掩模会按逻辑“或”合并。
- **阈值**：`--alpha_threshold` 控制全局水印的二值化阈值，范围 [0,1]，默认 0.5。
- **gating**：默认启用掩模 gating（除非 `--no_mask_gating`），训练和推理时会对输入图像执行 `image = image * (1 - mask)`，将水印区域像素置零，有助于模型专注于商品主体。

## 评测指标

验证阶段，脚本会输出以下指标：

- `top1`：按向量余弦相似度检索的 Top-1 召回率，衡量同款检索效果。
- `cat_acc`：类目分类准确率。
- `wm_acc`：水印分类准确率。

用户可根据具体业务需求调整损失权重，从而优化重要指标的表现。

## 后续扩展

* 采用更先进的 metric learning 损失，如 ArcFace、Circle Loss 等，以提升嵌入空间质量。
* 引入 cross-batch memory 或 queue 机制，缓解 batch-hard 在大规模数据上的效果下降。
* 替换骨干网络为更轻量或更强大的模型，如 EfficientNet、Swin Transformer。
* 使用在线难样本挖掘策略或增加数据增强，提高泛化能力。

## 注意事项

1. **水印文件存放**：建议将水印 PNG/PSD 文件存放在项目外部的数据目录，如 `data/watermarks/wm.png`，通过 `--global_watermark_path` 参数传入脚本。代码无需在仓库中附带该文件。
2. **PSD 支持**：当前脚本默认支持 PNG 格式。若使用 PSD，请先离线转换为 PNG 的带透明通道文件；或使用 `utils.py` 中的 `load_psd_mask` 函数，但需要安装 `psd-tools`。
3. 本项目为训练代码示例，无法保证在所有数据集上的最终精度。实际效果取决于标注质量、类目分布和超参调节。建议根据验证集指标调优。
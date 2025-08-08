vectorization_model_service
===========================

此模块是基于现有 1.0 版本的图片向量化服务改造而来，针对电商鞋包表珠宝类目开发的企业级向量化服务。核心目标是：

* **使用训练好的多任务模型生成稳定的向量嵌入**，支持水印掩模抑制，并根据类别进行裁剪，以提高不同类目之间的匹配精度；
* **保持业务接口不变**，兼容现有 Java→Python→Milvus 的链路。Python 服务专注于向量化，不直接操作 Milvus，让 Java 层负责入库和检索，减轻模型服务压力；
* **利用 Redis 缓存** 按图片内容生成的向量，避免重复计算，提高吞吐量；
* 提供可扩展的分类裁剪策略和人物抑制逻辑，后续可替换为更强的分割模型（例如 Mask R‑CNN）【89920199025526†L2104-L2112】【89920199025526†L2135-L2143】。

文件结构
--------

```
vectorization_model_service/
│
├── README.md            # 使用说明和项目概述
├── requirements.txt     # Python 依赖
├── config.py            # 服务配置项（模型路径、Redis配置、API密钥等）
├── model.py             # 多任务网络结构及权重加载函数
├── preprocess.py        # 预处理模块，包含人物抑制和按类目裁剪
├── service.py           # ImageEmbeddingService：封装模型推理、缓存逻辑、向量生成
└── app.py               # FastAPI 应用，提供接口入口
```

快速开始
--------

1. **安装依赖**：

   ```bash
   pip install -r requirements.txt
   ```

2. **准备模型权重**：将训练得到的 `model_final.pth` 放置在 config.py 指定的位置（默认 `./model_final.pth`）。

3. **启动服务**：

   ```bash
   uvicorn vectorization_model_service.app:app --reload --host 0.0.0.0 --port 8000
   ```

4. **请求示例**：

   向 `/v1/feature` 发送 POST 请求，Body 为 JSON，包含 `apiKey`（校验），`category`（可选）和 `imageBase64`（Base64 编码的图片）。示例：

   ```json
   {
     "apiKey": "your-secret-key",
     "category": "Shoes",
     "imageBase64": "iVBORw0KGgoAAAANSUhEUg..."
   }
   ```

   返回数据包含向量和水印概率：

   ```json
   {
     "code": 0,
     "message": "success",
     "vector": [0.1, 0.2, ...],
     "watermark_prob": 0.0012
   }
   ```

5. **集成链路**：

   业务链路保持不变：商家/用户上传图片 → Java 层调用该服务获取向量 → Java 层消费 Redis/消息队列将向量入库到 Milvus → 当用户检索时，Java 调用 Milvus 搜索并返回相似商品。此服务仅负责计算向量，不直接调用 Milvus，因而易于横向扩展。

说明
----

* **人物抑制与裁剪**：`preprocess.py` 使用 TorchVision 提供的 Faster R‑CNN 模型检测人像并遮挡，同时根据品类裁剪关键区域（例如鞋子保留底部 2/3，手表放大中心区域）。若模型无法加载，则退化为中心裁剪【89920199025526†L2104-L2112】【89920199025526†L2135-L2143】。
* **全局水印掩模**：配置项 `GLOBAL_WATERMARK_PATH` 支持提供水印 PNG（带透明通道），训练和推理时构建掩模，抑制水印区域对特征的影响。
* **向量缓存**：`service.py` 采用图片内容哈希为 Redis 键，向量以二进制格式缓存；再次请求相同图片时可直接返回缓存结果，提升效率。
* **API 密钥**：请求需携带正确的 `apiKey`，否则返回错误码 401。可在 `config.py` 中配置多个有效密钥。
* **分类裁剪扩展**：默认实现提供了鞋、包、手表、珠宝的简单裁剪策略，实际应用中可以根据训练数据调整函数或接入外部检测模型。

更多建议
--------

* **人物分割与杂物清理**：若对模特、手部等干扰需要更准确的分离，可以换用 `maskrcnn_resnet50_fpn` 等实例分割模型，对遮罩结果再叠加至特征提取层【89920199025526†L2104-L2112】。
* **分类专用 detector**：对于鞋子、包、手表等，可以训练或引入专用对象检测模型，如 YOLOv8、DINO 等，用于定位商品区域并裁剪，提高匹配精度。
* **异步处理与队列**：可将图片上传和向量生成拆分成生产-消费架构：上传 API 将 Base64 写入消息队列，由多个 Python 服务节点异步读取并生成向量，再由 Java 层消费入库。

# Model-Service
先将商品主图图片去白底、裁剪主体，有了识别的信息再对比详情图和sku图将向量更加丰富
## 运行
1. 本地：
   ```bash
   pip install -r requirements.txt
   uvicorn app:app --host 0.0.0.0 --port 5000
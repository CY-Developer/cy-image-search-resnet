# Image Search Service vFinal

## 构建 & 启动

```bash
docker build -t image-search-service .
docker run --gpus all -d -p 5000:5000 image-search-service

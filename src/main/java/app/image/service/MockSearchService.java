package app.image.service;

import app.image.entity.Product;
import app.image.entity.ProductItem;
import app.image.entity.ResultItem;
import app.image.mapper.ProductMapper;
import lombok.Data;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.io.IOException;
import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

@Service
public class MockSearchService {
    @Autowired
    private ProductMapper productMapper; // 新增：注入 CategoryMapper
    @Autowired
    private ImageSearchService imageSearchService; // 新增：注入 CategoryMapper
    @Autowired
    private LocalImageSearchClient localImageSearchClient; // 新增：注入 CategoryMapper


    public List<ProductItem> top20(MultipartFile image, int topk) throws Exception {
        List<ProductItem> items = new ArrayList<>();
//        ImageSearchService.SearchResponse searchResponse = imageSearchService.searchByImage(image, topk);
//        List<ImageSearchService.MergedHit> results = searchResponse.getResults();
//        List<Long> list = results.stream().map(ImageSearchService.MergedHit::getProductId).toList();
//        int[] array = {33995,33996,33997,33998,33999,34000,34001,34002,34003,34004,
//                34005,34006,34007,34008,34009,34010,34011,34012,34013,34014,34015};
//        List<Long> idList = java.util.stream.IntStream.of(array)
//                .mapToLong(i -> i)
//                .boxed()
//                .toList();
//        List<Product> products = productMapper.selectBatchIds(list);
        List<ResultItem> itemse = localImageSearchClient.searchTopK(storeFile(image), 10);
        itemse.forEach(product -> {
            // ===== 在这里替换为你真实的 20 个商品（示例演示）=====
            seed(items, String.valueOf(product.getProductId()), "",
                    product.getImagePath());
        });
        return items;
    }

    private void seed(List<ProductItem> items, String productId, String title, String catalogPath) {
        // 注意：不在这里转本地盘符，只拼一个后端可访问的图片 URL 给前端
        String encoded = URLEncoder.encode(catalogPath, StandardCharsets.UTF_8);

        String imageUrl = "/api/images?catalogPath=" + encoded; // 由后端 /api/images 读取本地并回传
        items.add(new ProductItem(productId, title, catalogPath, imageUrl));
    }
    // 从配置文件中读取文件存储目录（建议在application.properties中配置）
    @Value("${app.upload.dir:E:\\wrok\\app\\idea\\code\\cy-image-search-resnet\\images}")
    private String uploadDir;

    /**
     * 将MultipartFile转存到指定目录，并返回绝对路径
     */
    public String storeFile(MultipartFile file) throws IOException {
        // 校验文件是否为空
        if (file.isEmpty()) {
            throw new IllegalArgumentException("上传的文件不能为空");
        }

        // 确保上传目录存在
        File dir = new File(uploadDir);
        if (!dir.exists()) {
            dir.mkdirs(); // 递归创建目录（包括父目录）
        }

        // 生成唯一文件名（避免文件名重复覆盖）
        String originalFilename = file.getOriginalFilename();
        String fileExtension = originalFilename.substring(originalFilename.lastIndexOf("."));
        String uniqueFileName = UUID.randomUUID().toString() + fileExtension;

        // 创建目标文件对象
        File targetFile = new File(dir, uniqueFileName);

        // 转存文件（核心操作）
        file.transferTo(targetFile);

        // 返回绝对路径
        return targetFile.getAbsolutePath();
    }
}

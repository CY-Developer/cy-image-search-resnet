package app.image.service;

import app.image.entity.Product;
import app.image.entity.ProductItem;
import app.image.mapper.CategoryMapper;
import app.image.mapper.ProductMapper;
import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

@Service
public class MockSearchService {
    @Autowired
    private ProductMapper productMapper; // 新增：注入 CategoryMapper


    public List<ProductItem> top20() {
        List<ProductItem> items = new ArrayList<>();
        int[] array = {33995,33996,33997,33998,33999,34000,34001,34002,34003,34004,
                34005,34006,34007,34008,34009,34010,34011,34012,34013,34014,34015};
        List<Long> idList = java.util.stream.IntStream.of(array)
                .mapToLong(i -> i)
                .boxed()
                .toList();
        List<Product> products = productMapper.selectBatchIds(idList);
        products.forEach(product -> {
            // ===== 在这里替换为你真实的 20 个商品（示例演示）=====
            seed(items, String.valueOf(product.getProductId()), "",
                    product.getImage());
        });
        return items;
    }

    private void seed(List<ProductItem> items, String productId, String title, String catalogPath) {
        // 注意：不在这里转本地盘符，只拼一个后端可访问的图片 URL 给前端
        String encoded = URLEncoder.encode(catalogPath, StandardCharsets.UTF_8);
        String imageUrl = "/api/images?catalogPath=" + encoded; // 由后端 /api/images 读取本地并回传
        items.add(new ProductItem(productId, title, catalogPath, imageUrl));
    }
}

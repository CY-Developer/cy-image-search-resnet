package app.image.service;

import app.image.entity.ProductImages;
import app.image.entity.Product;
import app.image.entity.ProductImage;
import app.image.entity.ProductDescription;
import app.image.mapper.ProductMapper;
import app.image.mapper.ProductImageMapper;
import app.image.mapper.ProductDescriptionMapper;
import app.image.mapper.CategoryMapper;  // 新增
import app.image.utils.HtmlUtils;
import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import org.apache.commons.collections4.MapUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.*;

/**
 * 按月查询
 */
@Service
public class ImageInfoService {
    @Autowired
    private ProductMapper productMapper;
    @Autowired
    private ProductImageMapper productImageMapper;
    @Autowired
    private ProductDescriptionMapper productDescriptionMapper;

    @Autowired
    private CategoryMapper categoryMapper; // 新增：注入 CategoryMapper

    // 获取某月所有商品的所有图片，返回 ProductImages 列表
    public List<ProductImages> getAllProductImagesByMonth(int year, int month) {
        // 1. 一次查出本月所有主图商品
        QueryWrapper<Product> qw = new QueryWrapper<Product>()
                .apply("YEAR(date_added) = {0} AND MONTH(date_added) = {1}", year, month);
        List<Product> products = productMapper.selectList(qw);
        products.forEach(p -> {p.setImage(HtmlUtils.cutToSix(p.getImage()));});

        List<Long> productIds = products.stream().map(Product::getProductId).toList();

        // 2. 批量查询所有商品的类目ID（避免在循环中查询）
        List<Map<String, Object>> productCategories = categoryMapper.getCategoriesByProductIds(productIds);

        // 创建商品ID到类目名称的映射
        Map<Long, List<String>> productCategoryMap = new HashMap<>();
        for (Map<String, Object> productCategory : productCategories) {
            Long productId = MapUtils.getLong(productCategory,"product_id") ;
            String category = MapUtils.getString(productCategory,"category_name");

            productCategoryMap.computeIfAbsent(productId, k -> new ArrayList<>()).add(category);
        }

        // 将类目名称存入产品类
        for (Product product : products) {
            List<String> categorys = productCategoryMap.getOrDefault(product.getProductId(), new ArrayList<>());
            product.setCategory(String.join(", ", categorys));  // 将类目名称设置到产品类目字段
        }

        // 3. 一次查出本月所有附图
        QueryWrapper<ProductImage> imgQ = new QueryWrapper<>();
        imgQ.in("product_id", productIds);
        List<ProductImage> allAdditionalImages = productImageMapper.selectList(imgQ);
        Map<Long, List<String>> addMap = new HashMap<>();
        for (ProductImage img : allAdditionalImages) {
            img.setImage(HtmlUtils.cutToSix(img.getImage()));
            addMap.computeIfAbsent(img.getProductId(), k -> new ArrayList<>()).add(img.getImage());
        }

        // 4. 一次查出本月所有详情描述
        QueryWrapper<ProductDescription> descQ = new QueryWrapper<>();
        descQ.in("product_id", productIds);
        List<ProductDescription> allDesc = productDescriptionMapper.selectList(descQ);
        Map<Long, List<String>> detailMap = new HashMap<>();
        for (ProductDescription desc : allDesc) {
            if (desc.getDescription() != null) {
                detailMap.put(desc.getProductId(),
                        HtmlUtils.extractCatalogImagePaths(desc.getDescription()));
            }
        }

        // 5. 构建返回结果
        List<ProductImages> all = new ArrayList<>();
        for (Product p : products) {
            ProductImages pi = new ProductImages();
            pi.setProductId(p.getProductId());
            pi.setMainImage(p.getImage());
            pi.setAdditionalImages(addMap.getOrDefault(p.getProductId(), List.of()));
            pi.setDetailImages(detailMap.getOrDefault(p.getProductId(), List.of()));
            pi.setCategory(p.getCategory());  // 加上类目名称
            all.add(pi);
        }
        return all;
    }
}

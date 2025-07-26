package app.image.service;

import app.image.entity.ProductImages;
import app.image.entity.Product;
import app.image.entity.ProductImage;
import app.image.entity.ProductDescription;
import app.image.mapper.ProductMapper;
import app.image.mapper.ProductImageMapper;
import app.image.mapper.ProductDescriptionMapper;
import app.image.utils.HtmlUtils;
import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.*;

@Service
public class ImageInfoService {
    @Autowired
    private ProductMapper productMapper;
    @Autowired
    private ProductImageMapper productImageMapper;
    @Autowired
    private ProductDescriptionMapper productDescriptionMapper;

    // 获取某月所有商品的所有图片，返回 ProductImages 列表
    public List<ProductImages> getAllProductImagesByMonth(int year, int month) {
        // 1. 一次查出本月所有主图商品
        QueryWrapper<Product> qw = new QueryWrapper<Product>()
                .apply("YEAR(date_added) = {0} AND MONTH(date_added) = {1}", year, month);
        List<Product> products = productMapper.selectList(qw);
        products.forEach(p -> {p.setImage(HtmlUtils.cutToSix(p.getImage()));});
        List<Long> productIds = products.stream().map(Product::getProductId).toList();

        // 2. 一次查出本月所有附图
        QueryWrapper<ProductImage> imgQ = new QueryWrapper<>();
        imgQ.in("product_id", productIds);
        List<ProductImage> allAdditionalImages = productImageMapper.selectList(imgQ);
        Map<Long, List<String>> addMap = new HashMap<>();
        for (ProductImage img : allAdditionalImages) {
            img.setImage(HtmlUtils.cutToSix(img.getImage()));
            addMap.computeIfAbsent(img.getProductId(), k -> new ArrayList<>()).add(img.getImage());
        }

        // 3. 一次查出本月所有详情描述
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

        List<ProductImages> all = new ArrayList<>();
        for (Product p : products) {
            ProductImages pi = new ProductImages();
            pi.setProductId(p.getProductId());
            pi.setMainImage(p.getImage());
            pi.setAdditionalImages(addMap.getOrDefault(p.getProductId(), List.of()));
            pi.setDetailImages(detailMap.getOrDefault(p.getProductId(), List.of()));
            all.add(pi);
        }
        return all;
    }
}

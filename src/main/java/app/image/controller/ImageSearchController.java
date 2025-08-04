package app.image.controller;

import app.image.entity.ImageInfo;
import app.image.entity.ProductImages;
import app.image.service.ImageEmbeddingService;
import app.image.service.ImageInfoService;
import app.image.service.ImageSearchService;
import com.alibaba.fastjson.JSON;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * 图片搜索相关接口
 */
@Slf4j
@RestController
@RequestMapping("/api/search")
public class ImageSearchController {

    @Autowired
    private ImageEmbeddingService imageEmbeddingService;


    @Autowired
    private ImageInfoService imageInfoService;

    private static final String BASE_IMG_PATH = "model-service/6_jw7Ja";



    @Autowired
    private ImageSearchService imageSearchService;


    @PostMapping("/batch-import/search")
    public Map<String, Object> searchByImage(@RequestParam("file") MultipartFile file) {
        try {
            return Map.of("success", true, "message",imageSearchService.searchProducts(file));
        } catch (Exception e) {
            return Map.of("success", false, "message", e.getMessage());
        }
    }


    @PostMapping("/batch-import/month")
    public String importAll(@RequestParam int year, @RequestParam int month) {
        List<ProductImages> list = imageInfoService.getAllProductImagesByMonth(year, month);
        List<ProductImages> filterList = list.stream().filter(p -> !p.getCategory().isEmpty()).toList();
        int count = 0;
        for (ProductImages pi : filterList) {
            try {
                String taskId = imageEmbeddingService.extractBatchOneProduct(
                        BASE_IMG_PATH, pi.getProductId(), pi.getCategory(), pi.getMainImage(),
                            pi.getAdditionalImages(), pi.getDetailImages()
                );
                if (taskId != null && taskId.isEmpty()) {
                    count++;
                }
            } catch (Exception e) {
                log.error("Product {} import failed", pi.getProductId(), e);
            }
        }
        return "本月导入完毕, 成功商品数: " + count
                + "查出来的商品Id：" + JSON.toJSONString(list.stream().map(ProductImages::getProductId).toList())
                + "过滤掉没有类目商品的id" + JSON.toJSONString(filterList.stream().map(ProductImages::getProductId).toList());
    }
}

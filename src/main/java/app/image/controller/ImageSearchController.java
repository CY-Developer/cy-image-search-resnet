package app.image.controller;

import app.image.entity.ImageInfo;
import app.image.entity.ProductImages;
import app.image.service.ImageEmbeddingService;
import app.image.service.ImageInfoService;
import app.image.service.MilvusVectorService;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.util.ArrayList;
import java.util.List;

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
    private MilvusVectorService milvusVectorService;

    @Autowired
    private ImageInfoService imageInfoService;

    private static final String BASE_IMG_PATH = "model-service/6_jw7Ja";

//    /**
//     * 上传单张图片，返回最相似的20个商品ID
//     * @param file 图片文件
//     * @return 商品ID列表（按相似度排序）
//     */
//    @PostMapping(value = "/by-image", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
//    public List<Long> searchByImage(@RequestParam("file") MultipartFile file) {
//        try {
//            byte[] imgBytes = file.getBytes();
//            List<Float> vector = imageEmbeddingService.extractFromBytes(imgBytes);
//            return milvusVectorService.searchTopN(vector, 20);
//        } catch (Exception e) {
//            log.error("searchByImage error", e);
//            return new ArrayList<>();
//        }
//    }



    @PostMapping("/batch-import/month")
    public String importAll(@RequestParam int year, @RequestParam int month) {
        List<ProductImages> list = imageInfoService.getAllProductImagesByMonth(year, month);
        int count = 0;
        for (ProductImages pi : list) {
            try {
                List<Float> vector = imageEmbeddingService.extractBatchOneProduct(
                        BASE_IMG_PATH, pi.getProductId(), pi.getMainImage(), pi.getAdditionalImages(), pi.getDetailImages()
                );
                milvusVectorService.insert(pi.getProductId(), vector);
                count++;
            } catch (Exception e) {
                log.error("Product {} import failed", pi.getProductId(), e);
            }
        }
        return "本月导入完毕, 成功商品数: " + count;
    }
}

package app.image.service;

import org.springframework.core.io.FileSystemResource;
import org.springframework.core.io.Resource;
import org.springframework.http.*;
import org.springframework.stereotype.Service;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.client.RestTemplate;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

/**
 * 商品数据组装
 */

@Service
public class ImageEmbeddingService {
    private final RestTemplate rest = new RestTemplate();

    /**
     * 批量接口: 单个商品，多文件多字段上传
     */
    public String extractBatchOneProduct(String basePath, Long productId, String categoryName,  String mainImage, List<String> additionalImages, List<String> detailImages) throws Exception {
        HttpHeaders headers = new HttpHeaders();
        headers.set("X-API-Key", "93c1240be757f04a38c2aeb7e5cd7178");
        headers.setContentType(MediaType.MULTIPART_FORM_DATA);

        MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
        body.add("product_id", productId.toString());
        body.add("categoryName", categoryName);
        if (mainImage != null) {
            Path mainPath = Paths.get(basePath, mainImage);
            body.add("main_image", new FileSystemResource(mainPath.toFile()));
        }
        if (additionalImages != null) {
            for (String add : additionalImages) {
                if (add != null && !add.isEmpty()) {
                    Path p = Paths.get(basePath, add);
                    body.add("additional_images", new FileSystemResource(p.toFile()));
                }
            }
        }
        if (detailImages != null) {
            for (String detail : detailImages) {
                if (detail != null && !detail.isEmpty()) {
                    Path p = Paths.get(basePath, detail);
                    body.add("detail_images", new FileSystemResource(p.toFile()));
                }
            }
        }
        HttpEntity<MultiValueMap<String, Object>> requestEntity = new HttpEntity<>(body, headers);

        ResponseEntity<Map> response = rest.postForEntity("http://localhost:5000/extract-batch", requestEntity, Map.class);
        if (!response.getStatusCode().is2xxSuccessful() || response.getBody() == null) {
            throw new RuntimeException("Batch extract failed, productId=" + productId);
        }
        // 假设返回 JSON 里有 vector 字段
        return  String.valueOf(response.getBody().get("task_id"));
    }
}

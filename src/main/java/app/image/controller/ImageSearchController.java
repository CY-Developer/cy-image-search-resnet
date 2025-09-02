package app.image.controller;

import app.image.entity.ProductImages;
import app.image.entity.ProductItem;
import app.image.service.ImageEmbeddingService;
import app.image.service.ImageInfoService;
import app.image.service.MockSearchService;
import app.image.utils.HtmlUtils;
import com.alibaba.fastjson.JSON;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.*;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.server.ResponseStatusException;

import java.io.File;
import java.io.IOException;
import java.net.URLDecoder;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;

@Slf4j
@RestController
@RequestMapping("/api")
public class ImageSearchController {
    @Autowired
    private MockSearchService mockSearchService;
    @Autowired
    private ImageInfoService imageInfoService;
    @Autowired
    private ImageEmbeddingService imageEmbeddingService;

    private static final String BASE_IMG_PATH = "model-service/6_jw7Ja";
    // 缓存：catalogPath(截到/6/后) -> 本地绝对路径
    private final Map<String, Path> cache = new HashMap<>();

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

    @PostMapping(value = "/search", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public List<ProductItem> search(
            @RequestPart("image") MultipartFile image,
            @RequestParam(value = "topk", required = false, defaultValue = "10") int topk
    ) throws Exception {
        if (image == null || image.isEmpty()) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "image 不能为空");
        }

        return mockSearchService.top20(image,
                topk);
    }

    /**
     * 读取图片：把 catalog 路径转换到本地并输出二进制
     */
    @GetMapping("/images")
    public ResponseEntity<byte[]> image(@RequestParam("catalogPath") String catalogPath) throws IOException {
        String decoded = URLDecoder.decode(catalogPath, StandardCharsets.UTF_8);
        // 只保留 /6/ 以及之后的部分（你已有的工具方法）
        String sixTail = HtmlUtils.cutToSix("/" + decoded.replace("\\", "/"));
        if (sixTail.startsWith("/")) sixTail = sixTail.substring(1); // 去掉前导/

        Path local = cache.get(sixTail);
        if (local == null) {
            local = resolveLocal(sixTail); // 根据本地根目录 + 通配一层中间目录 去定位真实文件
            if (local != null) cache.put(sixTail, local);
        }

        if (local == null || !Files.exists(local)) {
            return ResponseEntity.status(HttpStatus.NOT_FOUND).body(new byte[0]);
        }

        // 读取并返回
        String ctype = Files.probeContentType(local);
        if (ctype == null) ctype = "image/jpeg";
        byte[] bytes = Files.readAllBytes(local);
        return ResponseEntity.ok()
                .contentType(MediaType.parseMediaType(ctype))
                .cacheControl(CacheControl.noCache())
                .body(bytes);
    }

    private Path resolveLocal(String sixTail) throws IOException {
        Path root = Paths.get("E:\\wrok\\app\\idea\\code\\cy-image-search-resnet\\model-service\\6_jw7Ja");

        // 方案1：直接拼
        Path p1 = root.resolve(sixTail.replace("/", File.separator));
        if (Files.exists(p1)) return p1;

        // 方案2：在根下一层尝试匹配
        try (DirectoryStream<Path> ds = Files.newDirectoryStream(root, Files::isDirectory)) {
            for (Path sub : ds) {
                Path p = sub.resolve(sixTail.replace("/", File.separator));
                if (Files.exists(p)) return p;
            }
        }
        return null;
    }
}
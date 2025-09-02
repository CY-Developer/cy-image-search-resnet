package app.image.service;

import com.fasterxml.jackson.annotation.JsonProperty;
import io.milvus.client.MilvusClient;
import io.milvus.grpc.SearchResults;
import io.milvus.param.R;
import io.milvus.param.dml.SearchParam;
import io.milvus.param.MetricType;
import io.milvus.response.SearchResultsWrapper;
import lombok.AllArgsConstructor;
import org.springframework.http.*;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.stereotype.Service;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.multipart.MultipartFile;
import shaded.parquet.it.unimi.dsi.fastutil.longs.LongArrayList;

import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.stream.Collectors;

/**
 * 新版向量检索服务：
 * 1. 支持分别用【裁剪向量】和【原图向量】检索并加权合并结果；
 * 2. 不对图片向量做融合，每张图一个向量；
 * 3. 返回格式可直接供 Redis 消费端 (RedisTaskConsumer) 使用。
 */
@Slf4j
@Service
public class ImageSearchService {

    // ======== 与 Milvus 表结构相关配置 ========
    @Value("${milvus.collection.name:product_vectors}")
    private String collection;

    @Value("${vector.dimension:2048}")
    private int vectorDim;

    /** 搜索结果集大小（每条向量检索返回的 topK） */
    @Value("${search.topk:50}")
    private int defaultTopK;

    /** 检索度量，可选：COSINE / IP / L2 */
    @Value("${search.metric:IP}")
    private String metric;

    /** 合并时裁剪向量权重 */
    @Value("${search.weight.crop:1.0}")
    private float weightCrop;

    /** 合并时原图向量权重 */
    @Value("${search.weight.full:0.85}")
    private float weightFull;

    /** 检索得分过滤阈值 */
    @Value("${search.score.threshold:0.0}")
    private float scoreThreshold;

    @Value("${vector.python.base-url}")
    private String PY_BASE;

    @Value("${vector.python.api-key}")
    private String PY_KEY;

    @Autowired
    private RestTemplate restTemplate;

    private final MilvusClient milvus;

    public ImageSearchService(MilvusClient milvusClient) {
        this.milvus = milvusClient;
    }

    /**
     * 入口A：接收裁剪向量和原图向量，返回检索结果。
     * 向量长度应为 vectorDim（512）。
     *
     * @param cropVec 裁剪后图片的向量 (可以为 null)
     * @param fullVec 原图的向量 (可以为 null)
     * @param topK 搜索返回条数
     * @return SearchResponse
     */
    public SearchResponse searchByVectors(List<Float> cropVec, List<Float> fullVec, Integer topK) {
        int tk = (topK == null || topK <= 0) ? defaultTopK : topK;
        Map<Long, MergedHit> merged = new HashMap<>();

        MetricType mt = parseMetric(metric);

        if (validVec(cropVec)) {
            List<Float> q = MetricType.IP.equals(mt) ? l2Normalize(cropVec) : cropVec;
            accumulate(merged, searchOnce(q, tk), weightCrop);
        }
        if (validVec(fullVec)) {
            List<Float> q = MetricType.IP.equals(mt) ? l2Normalize(fullVec) : fullVec;
            accumulate(merged, searchOnce(q, tk), weightFull);
        }


        // 过滤并按分数排序
        List<MergedHit> sorted = merged.values().stream()
                .filter(h -> h.score >= scoreThreshold)
                .sorted(Comparator.comparingDouble((MergedHit h) -> -h.score))
                .limit(tk)
                .collect(Collectors.toList());

        return new SearchResponse(sorted);
    }
    private List<Float> l2Normalize(List<Float> v) {
        double sum = 0.0;
        for (Float x : v) sum += x * x;
        double norm = Math.sqrt(sum);
        if (norm <= 0) return v;
        List<Float> out = new ArrayList<>(v.size());
        for (Float x : v) out.add((float)(x / norm));
        return out;
    }

    /**
     * 入口B：接收图片文件，内部处理裁剪与原图向量（此处仅示例，可按实际替换）。
     * 实际项目中建议在Controller或其它服务层先调用Python向量服务得到向量。
     *
     * @param image 上传图片
     * @param topK  返回条数
     * @return SearchResponse
     * @throws Exception 读取图片失败
     */
    public SearchResponse searchByImage(MultipartFile image, Integer topK) throws Exception {
        byte[] bytes = image.getBytes();
        // stub：请替换为真实的向量获取逻辑
        EmbedQueryResp resp = fetchVectorsFromPython(bytes);
        List<Float> cropVec = resp.vectorCrop;
        List<Float> fullVec = resp.vectorFull;
        return searchByVectors(cropVec, fullVec, topK);
    }

    // ======== 替换 stub：真正调用 Python /embed-query ========
    private EmbedQueryResp fetchVectorsFromPython(byte[] bytes) {
        String url = PY_BASE + "/embed-query";

        // 用 ByteArrayResource 作为文件 part
        ByteArrayResource filePart = new ByteArrayResource(bytes) {
            @Override public String getFilename() { return "query.jpg"; }
        };

        MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
        body.add("image", filePart);

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.MULTIPART_FORM_DATA);
        headers.set("X-API-Key", PY_KEY);

        HttpEntity<MultiValueMap<String, Object>> req = new HttpEntity<>(body, headers);


        ResponseEntity<EmbedQueryResp> resp = restTemplate.exchange(
                url, HttpMethod.POST, req, EmbedQueryResp.class);

        if (!resp.getStatusCode().is2xxSuccessful() || resp.getBody() == null) {
            throw new RuntimeException("embed-query 调用失败：" + resp.getStatusCode());
        }
        if (!"ok".equalsIgnoreCase(resp.getBody().status)) {
            throw new RuntimeException("embed-query 返回错误：" + resp.getBody().status);
        }
        // sanity check：校验维度
        if (resp.getBody().vectorFull != null && resp.getBody().vectorFull.size() != vectorDim) {
            throw new RuntimeException("full 向量维度不匹配");
        }
        if (resp.getBody().vectorCrop != null && resp.getBody().vectorCrop.size() != vectorDim) {
            // 允许裁剪失败 -> null；但如果非空又不匹配维度，视为异常
            throw new RuntimeException("crop 向量维度不匹配");
        }
        return resp.getBody();
    }

    // ======== Python 响应 DTO ========
    @Data
    public static class EmbedQueryResp {
        public String status;

        @JsonProperty("vector_full")
        public List<Float> vectorFull;

        @JsonProperty("vector_crop")
        public List<Float> vectorCrop;
    }

    // =================== Milvus 搜索核心实现 ===================

    /**
     * 只用一个向量搜索 Milvus
     *
     * @param vec 向量
     * @param topK 返回条数
     * @return RawHit 列表
     */
    private List<RawHit> searchOnce(List<Float> vec, int topK) {
        try {
            if (!validVec(vec)) return Collections.emptyList();
            SearchParam searchParam = SearchParam.newBuilder()
                    .withCollectionName(collection)
                    .withMetricType(parseMetric(metric))
                    .withOutFields(Arrays.asList("vector_id", "product_id", "image_type"))
                    .withTopK(topK)
                    .withVectors(Collections.singletonList(vec))
                    .withVectorFieldName("embedding")
                    .build();
            R<SearchResults> r = milvus.search(searchParam);
            if (r.getStatus() != R.Status.Success.getCode() || r.getData() == null) {
                log.warn("Milvus search failed: {}", r.getMessage());
                return Collections.emptyList();
            }
            SearchResultsWrapper wrapper = new SearchResultsWrapper(r.getData().getResults());
            List<SearchResultsWrapper.IDScore> idScoreList = wrapper.getIDScore(0);

            // 取回外字段
            List<Object> productIds = wrapper.getFieldWrapper("product_id").getFieldData().stream().collect(Collectors.toUnmodifiableList());
            List<Object> imageTypes = wrapper.getFieldWrapper("image_type").getFieldData().stream().collect(Collectors.toUnmodifiableList());
            List<Object> vectorIds  = wrapper.getFieldWrapper("vector_id").getFieldData().stream().collect(Collectors.toUnmodifiableList());

            List<RawHit> hits = new ArrayList<>();
            for (int i = 0; i < idScoreList.size(); i++) {
                float score = idScoreList.get(i).getScore();
                Long pid = parseLongSafe(productIds, i);
                String imageType = parseStringSafe(imageTypes, i);
                String vectorId  = parseStringSafe(vectorIds, i);
                if (pid == null) continue;
                hits.add(new RawHit(pid, vectorId, imageType, score));
            }
            return hits;
        } catch (Exception e) {
            log.error("searchOnce error", e);
            return Collections.emptyList();
        }
    }

    // =================== 合并 & 工具 ===================

    private void accumulate(Map<Long, MergedHit> sink, List<RawHit> hits, float weight) {
        for (RawHit h : hits) {
            MergedHit m = sink.computeIfAbsent(h.productId,
                    k -> new MergedHit(h.productId, h.vectorId, h.imageType, 0f));
            // 如果已有相同 productId，则累积权重后的分数，取最大值
            m.score = Math.max(m.score, h.score * weight);
        }
    }

    private boolean validVec(List<Float> v) {
        return v != null && v.size() == vectorDim && v.stream().allMatch(Objects::nonNull);
    }

    private MetricType parseMetric(String m) {
        if (m == null) return MetricType.IP;
        return switch (m.toUpperCase(Locale.ROOT)) {
            case "IP" -> MetricType.IP;
            case "L2" -> MetricType.L2;
            default -> MetricType.COSINE;
        };
    }

    private Long parseLongSafe(List<Object> arr, int idx) {
        try {
            return (arr == null) ? null : Long.parseLong(String.valueOf((arr.get(idx))));
        } catch (Exception e) {
            return null;
        }
    }

    private String parseStringSafe(List<Object> arr, int idx) {
        try { return (arr == null) ? null : String.valueOf(( arr.get(idx))); }
        catch (Exception e) { return null; }
    }

    // =================== 向量获取 stub（务必替换） ===================

    /**
     * 示例向量生成函数，请替换为实际的向量获取逻辑。
     */
    private List<Float> stubVectorize(byte[] data, boolean crop) {
        List<Float> v = new ArrayList<>(vectorDim);
        // 示例返回 512 维零向量，实际应调用Python服务获取裁剪和非裁剪向量
        for (int i = 0; i < vectorDim; i++) v.add(0f);
        return v;
    }

    // =================== DTO 定义 ===================

    @Data @NoArgsConstructor @AllArgsConstructor
    public static class SearchResponse {
        private List<MergedHit> results;
    }

    @Data @NoArgsConstructor @AllArgsConstructor
    public static class MergedHit {
        private Long productId;
        private String vectorId;   // 用于排查/回显，无需参与排序
        private String imageType;  // main/additional_x/detail_x
        private float score;       // 合并后的得分（已权重）
    }

    @Data @NoArgsConstructor @AllArgsConstructor
    public static class RawHit {
        private Long   productId;
        private String vectorId;
        private String imageType;
        private float  score;
    }
}

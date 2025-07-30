package app.image.service;

import com.alibaba.fastjson.JSON;
import com.alibaba.fastjson.TypeReference;
import io.milvus.client.MilvusClient;
import io.milvus.grpc.MutationResult;
import io.milvus.param.R;
import io.milvus.param.dml.DeleteParam;
import io.milvus.param.dml.InsertParam;
import io.milvus.param.dml.QueryParam;
import io.milvus.response.QueryResultsWrapper;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;

import java.time.Duration;
import java.util.*;
import java.util.stream.Collectors;

@Slf4j
@Service
public class RedisTaskConsumer {

    @Value("${milvus.collection.name:product_vectors}")
    private String COLLECTION;

    @Value("${vector.dimension:864}")
    private int VECTOR_DIM;

    private static final String FIELD_VECTOR_ID = "vector_id";
    private static final String FIELD_PRODUCT_ID = "product_id";
    private static final String FIELD_IMAGE_TYPE = "image_type";
    private static final String FIELD_EMBEDDING = "embedding";

    @Autowired
    private RedisTemplate<String, String> redisTemplate;

    @Autowired
    private MilvusClient milvusClient;

    @Scheduled(fixedDelay = 2000)
    public void consumeAllTasks() {
        Set<String> keys = redisTemplate.keys("task:*:res");
        if (keys == null || keys.isEmpty()) return;

        for (String key : keys) {
            try {
                boolean success = processBatch(key);
                if (success) {
                    redisTemplate.delete(key);
                } else {
                    redisTemplate.expire(key, Duration.ofHours(1));
                }
            } catch (Exception e) {
                log.error("Exception while processing task key: " + key, e);
            }
        }
    }

    private boolean processBatch(String key) {
        String jsonString = redisTemplate.opsForValue().get(key);
        if (jsonString == null) return false;

        Map<String, Object> obj = JSON.parseObject(jsonString, new TypeReference<>() {});
        Object recognizedRaw = obj.get("recognized");
        if (!(recognizedRaw instanceof List<?> recognizedList)) return false;

        List<String> vectorIds = new ArrayList<>();
        List<Long> productIds = new ArrayList<>();
        List<List<Float>> vectors = new ArrayList<>();
        List<String> imageTypes = new ArrayList<>();
        Set<Long> idsToDelete = new HashSet<>();

        for (Object item : recognizedList) {
            if (!(item instanceof Map<?, ?> recognized)) continue;

            Long productId = tryParseLong(recognized.get("product_id"));
            if (productId == null) continue;
            idsToDelete.add(productId);

            List<Float> mainVector = parseVector(recognized.get("vector"));
            if (mainVector == null) continue;

            vectorIds.add(genVectorId(productId, "main"));
            productIds.add(productId);
            vectors.add(mainVector);
            imageTypes.add("main");

            Object additional = recognized.get("vectors");
            if (additional instanceof List<?> additionalVectors) {
                for (int i = 0; i < additionalVectors.size(); i++) {
                    List<Float> vec = parseVector(additionalVectors.get(i));
                    if (vec == null) continue;
                    vectorIds.add(genVectorId(productId, "additional_" + i));
                    productIds.add(productId);
                    vectors.add(vec);
                    imageTypes.add("additional_" + i);
                }
            }
        }

        if (vectorIds.size() != productIds.size() || vectors.size() != productIds.size() || imageTypes.size() != productIds.size()) {
            log.warn("Field size mismatch for task {}: vectorIds={}, productIds={}, vectors={}, imageTypes={}",
                    key, vectorIds.size(), productIds.size(), vectors.size(), imageTypes.size());
            return false;
        }

        // 删除旧向量
        List<String> vectorIdsToDelete = queryVectorIdsByProductIds(new ArrayList<>(idsToDelete));
        batchDeleteByVectorIds(vectorIdsToDelete);

        // 插入新向量
        boolean success = batchInsertVectors(vectorIds, productIds, vectors, imageTypes);
        logFailedProducts(obj);
        return success;
    }

    private List<String> queryVectorIdsByProductIds(List<Long> productIds) {
        List<String> vectorIds = new ArrayList<>();
        for (Long pid : productIds) {
            try {
                QueryParam queryParam = QueryParam.newBuilder()
                        .withCollectionName(COLLECTION)
                        .withExpr(FIELD_PRODUCT_ID + " == " + pid)
                        .withOutFields(Collections.singletonList(FIELD_VECTOR_ID))
                        .build();

                R<io.milvus.grpc.QueryResults> result = milvusClient.query(queryParam);
                if (result.getStatus() != R.Status.Success.getCode()) {
                    log.warn("Query failed for product_id {}: {}", pid, result.getMessage());
                    continue;
                }

                QueryResultsWrapper wrapper = new QueryResultsWrapper(result.getData());
                List<Object> data = (List<Object>) wrapper.getFieldWrapper(FIELD_VECTOR_ID).getFieldData();
                for (Object idObj : data) {
                    if (idObj != null) vectorIds.add(idObj.toString());
                }
            } catch (Exception e) {
                log.warn("Query error for product_id {}: {}", pid, e.getMessage());
            }
        }
        return vectorIds;
    }

    private void batchDeleteByVectorIds(List<String> vectorIds) {
        if (vectorIds.isEmpty()) return;
        String expr = FIELD_VECTOR_ID + " in [" + vectorIds.stream()
                .map(id -> "\"" + id + "\"")
                .collect(Collectors.joining(",")) + "]";
        DeleteParam param = DeleteParam.newBuilder()
                .withCollectionName(COLLECTION)
                .withExpr(expr)
                .build();
        R<MutationResult> result = milvusClient.delete(param);
        if (!result.getStatus().equals(R.Status.Success.getCode())) {
            log.error("Delete failed: {}", result.getMessage());
        }
    }

    private boolean batchInsertVectors(List<String> vectorIds, List<Long> productIds,
                                       List<List<Float>> vectors, List<String> types) {
        if (vectorIds.isEmpty()) return false;
        InsertParam param = InsertParam.newBuilder()
                .withCollectionName(COLLECTION)
                .withFields(Arrays.asList(
                        new InsertParam.Field(FIELD_VECTOR_ID, vectorIds),
                        new InsertParam.Field(FIELD_PRODUCT_ID, productIds),
                        new InsertParam.Field(FIELD_IMAGE_TYPE, types),
                        new InsertParam.Field(FIELD_EMBEDDING, vectors)
                ))
                .build();
        R<MutationResult> result = milvusClient.insert(param);
        if (!result.getStatus().equals(R.Status.Success.getCode())) {
            log.error("Insert failed: {}", result.getMessage());
            return false;
        }
        return true;
    }

    private List<Float> parseVector(Object raw) {
        if (!(raw instanceof List<?> rawList)) return null;
        if (rawList.size() != VECTOR_DIM) {
            log.warn("向量维度不匹配，期望 {}, 实际 {}", VECTOR_DIM, rawList.size());
            return null;
        }
        try {
            return rawList.stream()
                    .map(Object::toString)
                    .map(Float::parseFloat)
                    .collect(Collectors.toList());
        } catch (Exception e) {
            log.warn("向量解析失败: {}", rawList, e);
            return null;
        }
    }

    private String genVectorId(Long productId, String type) {
        return productId + "_" + type + "_" + System.currentTimeMillis();
    }

    private Long tryParseLong(Object val) {
        try {
            return Long.parseLong(String.valueOf(val));
        } catch (Exception e) {
            return null;
        }
    }

    private void logFailedProducts(Map<String, Object> obj) {
        Object failed = obj.get("failed");
        if (failed != null) {
            log.info("Failed products: {}", JSON.toJSONString(failed));
        }
    }
}

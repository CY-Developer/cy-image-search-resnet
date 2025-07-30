package app.image.service;

import io.milvus.client.MilvusClient;
import io.milvus.grpc.SearchResults;
import io.milvus.param.R;
import io.milvus.param.dml.SearchParam;
import io.milvus.response.SearchResultsWrapper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;

@Service
public class ImageSearchService {

    private static final String COLLECTION = "product_vectors";
    private static final String VECTOR_FIELD = "embedding";
    private static final String PRODUCT_ID_FIELD = "product_id";
    private static final String IMAGE_TYPE_FIELD = "image_type";
    private static final int MAIN_TOP_K = 30;
    private static final int DETAIL_TOP_K = 5;
    private static final int ADDITIONAL_SUPPLY_K = 50;

    @Autowired
    private MilvusClient milvusClient;

    @Autowired
    private PythonVectorClient pythonVectorClient;

    public Map<String, Object> search(MultipartFile multipartFile) throws Exception {
        File tmp = File.createTempFile("upload-", ".jpg");
        multipartFile.transferTo(tmp);
        List<Float> queryVector = pythonVectorClient.extractVector(tmp);
        Map<String, Object> result = new LinkedHashMap<>();
        try {
            // 强制转换类型为 Float，防止 BigDecimal 报错
            List<?> rawVector = pythonVectorClient.extractVector(tmp);

            List<Float> fixedVector = rawVector.stream()
                    .map(v -> ((Number) v).floatValue())
                    .map(Float::valueOf)
                    .collect(Collectors.toList());

            // 1. 主图召回
            List<Long> mainCandidates = searchByImageType(fixedVector, "main", MAIN_TOP_K);

            // 2. 主图召回不足时，用附图/详情图补充
            Set<Long> candidates = new HashSet<>(mainCandidates);
            if (candidates.size() < DETAIL_TOP_K) {
                List<Long> additional = searchByImageType(fixedVector, null, ADDITIONAL_SUPPLY_K);
                candidates.addAll(additional);
            }

            // 3. 在所有候选商品下，查所有商品的附图/详情图做最终比对
            List<Map<String, Object>> detailResults = searchWithinCandidates(fixedVector, candidates, DETAIL_TOP_K);

            result.put("success", true);
            result.put("candidates", candidates);
            result.put("results", detailResults);

        } finally {
            if (tmp.exists()) tmp.delete();
        }
        return result;
    }

    private List<Long> searchByImageType(List<Float> fixedVector, String imageType, int topK) {
        String expr = (imageType != null)
                ? IMAGE_TYPE_FIELD + " == \"" + imageType + "\""
                : null;

        SearchParam.Builder builder = SearchParam.newBuilder()
                .withCollectionName(COLLECTION)
                .withOutFields(Arrays.asList(PRODUCT_ID_FIELD, IMAGE_TYPE_FIELD))
                .withTopK(topK)
                .withVectors(Collections.singletonList(fixedVector))
                .withVectorFieldName(VECTOR_FIELD);
        if (expr != null) {
            builder.withExpr(expr);
        }

        R<SearchResults> result = milvusClient.search(builder.build());
        if (!result.getStatus().equals(R.Status.Success.getCode())) {
            return Collections.emptyList();
        }
        SearchResultsWrapper wrapper = new SearchResultsWrapper(result.getData().getResults());
        List<Object> pidObjects = Collections.singletonList(wrapper.getFieldWrapper(PRODUCT_ID_FIELD).getFieldData());

        return pidObjects.stream()
                .filter(Objects::nonNull)
                .flatMap(obj -> {
                    if (obj instanceof Number) {
                        return Stream.of(((Number) obj).longValue());
                    } else if (obj instanceof String) {
                        try {
                            return Stream.of(Long.parseLong((String) obj));
                        } catch (Exception e) {
                            return Stream.empty();
                        }
                    } else if (obj instanceof List<?>) {
                        return ((List<?>) obj).stream()
                                .filter(Objects::nonNull)
                                .filter(o -> o instanceof Number)
                                .map(o -> ((Number) o).longValue());
                    } else {
                        return Stream.empty();
                    }
                })
                .distinct()
                .collect(Collectors.toList());
    }

    private List<Map<String, Object>> searchWithinCandidates(List<Float> fixedVector, Set<Long> productIds, int topK) {
        if (productIds == null || productIds.isEmpty()) return Collections.emptyList();

        String expr = PRODUCT_ID_FIELD + " in [" +
                productIds.stream().map(String::valueOf).collect(Collectors.joining(",")) +
                "] and " + IMAGE_TYPE_FIELD + " != \"main\"";

        SearchParam param = SearchParam.newBuilder()
                .withCollectionName(COLLECTION)
                .withOutFields(Arrays.asList(PRODUCT_ID_FIELD, IMAGE_TYPE_FIELD))
                .withExpr(expr)
                .withTopK(topK)
                .withVectors(Collections.singletonList(fixedVector))
                .withVectorFieldName(VECTOR_FIELD)
                .build();

        R<SearchResults> result = milvusClient.search(param);
        if (!result.getStatus().equals(R.Status.Success.getCode())) {
            return Collections.emptyList();
        }

        SearchResultsWrapper wrapper = new SearchResultsWrapper(result.getData().getResults());
        List<Object> productIdsResult = (List<Object>) wrapper.getFieldWrapper(PRODUCT_ID_FIELD).getFieldData();
        List<Object> imageTypes = (List<Object>) wrapper.getFieldWrapper(IMAGE_TYPE_FIELD).getFieldData();
        List<Float> scores = result.getData().getResults().getScoresList();

        List<Map<String, Object>> allResults = new ArrayList<>();
        for (int i = 0; i < scores.size(); i++) {
            Map<String, Object> row = new HashMap<>();
            row.put("product_id", productIdsResult.get(i));
            row.put("image_type", imageTypes.get(i));
            row.put("score", scores.get(i));
            allResults.add(row);
        }

        return allResults.stream()
                .sorted((a, b) -> Float.compare((Float) b.get("score"), (Float) a.get("score")))
                .limit(topK)
                .collect(Collectors.toList());
    }
}

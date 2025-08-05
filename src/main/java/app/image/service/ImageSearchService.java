package app.image.service;

import io.milvus.client.MilvusClient;
import io.milvus.grpc.SearchResults;
import io.milvus.grpc.SearchResultData;
import io.milvus.param.MetricType;
import io.milvus.param.R;
import io.milvus.param.dml.SearchParam;
import io.milvus.response.SearchResultsWrapper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.util.*;
import java.util.stream.Collectors;

@Service
public class ImageSearchService {

    @Autowired
    private PythonVectorClient pVectorClient;

    @Autowired
    private MilvusClient milvusClient;

    private static final String COLLECTION_NAME = "product_vectors";
    private static final String VECTOR_FIELD = "embedding";
    private static final int TOP_K_VECTOR = 100;
    private static final int TOP_N_PRODUCT = 10;

    public SearchParam buildSearchParam(String collectionName, String vectorField, List<?> queryVector, int topK) {
        List<Float> fixedVector = queryVector.stream()
                .map(v -> {
                    if (v instanceof Number) {
                        return ((Number) v).floatValue();
                    } else {
                        throw new IllegalArgumentException("向量必须是数字类型，实际类型: " + v.getClass().getName());
                    }
                })
                .collect(Collectors.toList());

        return SearchParam.newBuilder()
                .withCollectionName(collectionName)
                .withVectorFieldName(vectorField)
                .withOutFields(Collections.singletonList("product_id"))
                .withTopK(topK)
                .withMetricType(MetricType.IP) // 注意：必须和collection创建时metric一致
                .withVectors(Collections.singletonList(fixedVector))
                .withParams("{\"nprobe\":10}")
                .build();
    }

    public List<Map<String, Object>> rankByProductId(SearchResultsWrapper wrapper, SearchResultData resultData, int topN) {
        // 从wrapper获取product_id字段列表
        List<?> productIds = wrapper.getFieldData("product_id", 0);
        // 从resultData获取对应的分数列表
        List<Float> scoreList = resultData.getScoresList();

        if (productIds == null || scoreList == null || productIds.size() != scoreList.size()) {
            throw new IllegalStateException("Milvus 返回数据维度不一致");
        }

        // 按product_id聚合分数
        Map<String, List<Float>> grouped = new HashMap<>();
        for (int i = 0; i < productIds.size(); i++) {
            String pid = productIds.get(i).toString();
            grouped.computeIfAbsent(pid, k -> new ArrayList<>()).add(scoreList.get(i));
        }

        // 计算平均分，按分数倒序排序，返回前topN
        return grouped.entrySet().stream()
                .map(e -> {
                    Map<String, Object> map = new HashMap<>();
                    map.put("product_id", e.getKey());
                    map.put("score", average(e.getValue()));
                    return map;
                })
                .sorted((a, b) -> Float.compare((float) b.get("score"), (float) a.get("score")))
                .limit(topN)
                .collect(Collectors.toList());
    }

    private float average(List<Float> values) {
        return (float) values.stream().mapToDouble(Float::doubleValue).average().orElse(0.0);
    }

    public List<Map<String, Object>> searchProducts(MultipartFile multipartFile) throws Exception {
        File imageFile = File.createTempFile("upload-", ".jpg");
        multipartFile.transferTo(imageFile);

        List<?> queryVector = pVectorClient.extractVector(imageFile);

        SearchParam param = buildSearchParam(COLLECTION_NAME, VECTOR_FIELD, queryVector, TOP_K_VECTOR);
        R<SearchResults> result = milvusClient.search(param);

        if (result.getStatus() != R.Status.Success.getCode()) {
            throw new RuntimeException("Milvus 查询失败: " + result.getException().getMessage());
        }

        SearchResultData resultData = result.getData().getResults();
        SearchResultsWrapper wrapper = new SearchResultsWrapper(resultData);

        return rankByProductId(wrapper, resultData, TOP_N_PRODUCT);
    }
}

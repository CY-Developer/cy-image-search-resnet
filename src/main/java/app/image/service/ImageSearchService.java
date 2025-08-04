package app.image.service;

import io.milvus.client.MilvusServiceClient;
import io.milvus.client.MilvusClient;
import io.milvus.grpc.SearchResults;
import io.milvus.grpc.SearchResultData;
import io.milvus.param.ConnectParam;
import io.milvus.param.MetricType;
import io.milvus.param.R;
import io.milvus.param.dml.SearchParam;
import io.milvus.response.SearchResultsWrapper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.io.IOException;
import java.math.BigDecimal;
import java.util.*;
import java.util.stream.Collectors;

@Service
public class ImageSearchService {

    @Autowired
    private PythonVectorClient pVectorClient;
    @Autowired
    private MilvusClient milvusClient;
    // 常量参数
    private static final String COLLECTION_NAME = "product_vectors";
    private static final String VECTOR_FIELD = "embedding";
    private static final int TOP_K_VECTOR = 400;   // 向量库中返回前40张最相似的图片
    private static final int TOP_N_PRODUCT = 100;  // 聚类后最终返回TopN商品（按需过滤）

    private static final String MILVUS_HOST = "127.0.0.1";
    private static final int MILVUS_PORT = 19530;


    /**
     * 调用 Milvus 进行向量搜索
     */
    public SearchParam buildSearchParam(String collectionName, String vectorField, List<?> queryVector, int topK) {

        List<Float> fixedVector = queryVector.stream()
                .map(v -> {
                    if (v instanceof BigDecimal) {
                        return ((BigDecimal) v).floatValue();
                    } else if (v instanceof Number) {
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
                .withMetricType(MetricType.L2)
                .withVectors(Collections.singletonList(fixedVector))
                .withParams("{\"nprobe\":10}")
                .build();
    }

    /**
     * 聚合搜索结果（最终修正，完全兼容 Milvus 2.2.11）
     */

    public List<Map<String, Object>> rankByProductId(SearchResultsWrapper wrapper, SearchResultData resultData, int topN) {
        List<?> productIds = wrapper.getFieldData("product_id", 0);
        List<Float> scoreList = resultData.getScoresList();

        if (productIds == null || scoreList == null || productIds.size() != scoreList.size()) {
            throw new IllegalStateException("Milvus 返回数据维度不一致");
        }

        Map<String, List<Float>> grouped = new HashMap<>();
        for (int i = 0; i < productIds.size(); i++) {
            String pid = productIds.get(i).toString();
            grouped.computeIfAbsent(pid, k -> new ArrayList<>()).add(scoreList.get(i));
        }

        // 按平均分排序，并组装成你要的格式
        return grouped.entrySet().stream()
                .map(e -> {
                    Map<String, Object> map = new HashMap<>();
                    map.put("product_id", e.getKey());
                    map.put("score", average(e.getValue()));
                    return map;
                })
                .sorted((a, b) -> Float.compare(
                        (float) b.get("score"),
                        (float) a.get("score")
                ))
                .limit(topN)
                .collect(Collectors.toList());
    }


    private float average(List<Float> values) {
        return (float) values.stream().mapToDouble(Float::doubleValue).average().orElse(0.0);
    }

    /**
     * 主入口：上传图 → 提向量 → 查 Milvus → 聚类返回 TopN 商品ID（供商品过滤）
     */
    public  List<Map<String, Object>>  searchProducts(MultipartFile multipartFile) throws Exception {
        File imageFile = File.createTempFile("upload-", ".jpg");
        multipartFile.transferTo(imageFile);
        List<?> queryVector = pVectorClient.extractVector(imageFile);
        SearchParam param = buildSearchParam(COLLECTION_NAME, VECTOR_FIELD, queryVector, TOP_K_VECTOR);
        R<SearchResults> result = milvusClient.search(param);
        SearchResultData resultData = result.getData().getResults();
        SearchResultsWrapper wrapper = new SearchResultsWrapper(resultData);
        return rankByProductId(wrapper, resultData, TOP_N_PRODUCT);
    }
}

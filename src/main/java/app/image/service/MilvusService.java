package app.image.service;

import io.milvus.client.MilvusClient;
import io.milvus.grpc.SearchResults;
import io.milvus.param.R;
import io.milvus.param.dml.SearchParam;
import io.milvus.param.MetricType;
import io.milvus.response.SearchResultsWrapper;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import app.image.entity.SearchResult;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;

@Slf4j
@Service
public class MilvusService {

    private final MilvusClient milvusClient;

    @Value("${milvus.collection.name:product_vectors}")
    private String collectionName;

    public MilvusService(MilvusClient milvusClient) {
        this.milvusClient = milvusClient;
    }

    public List<SearchResult> search(float[] queryVector, int topK) {
        try {
            SearchParam searchParam = SearchParam.newBuilder()
                    .withCollectionName(collectionName)
                    .withMetricType(MetricType.IP)
                    .withOutFields(List.of("product_id"))
                    .withTopK(topK)
                    .withVectors(List.of(queryVector))
                    .withVectorFieldName("embedding")
                    .withParams("{\"nprobe\": 16}")
                    .build();

            R<SearchResults> result = milvusClient.search(searchParam);
            if (result.getStatus() != R.Status.Success.getCode()) {
                log.warn("Milvus search failed: {}", result.getMessage());
                return List.of();
            }

            SearchResultsWrapper wrapper = new SearchResultsWrapper(result.getData().getResults());

            List<Object> productIds = Collections.singletonList(wrapper.getFieldWrapper("product_id").getFieldData());
            List<Object> scoreObjs = Collections.singletonList(wrapper.getFieldWrapper("score").getFieldData());
            List<Float> scores = scoreObjs.stream()
                    .map(o -> (Float) o)
                    .collect(Collectors.toList());

            List<SearchResult> results = new ArrayList<>();
            for (int i = 0; i < productIds.size(); i++) {
                String productId = String.valueOf(productIds.get(i));
                float score = scores.get(i);
                results.add(new SearchResult(productId, score));
            }

            return results.stream()
                    .sorted(Comparator.comparingDouble(SearchResult::getScore).reversed())
                    .collect(Collectors.toList());

        } catch (Exception e) {
            log.error("Milvus search error", e);
            return List.of();
        }
    }


}

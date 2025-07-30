package app.image.service;

import io.milvus.client.MilvusClient;
import io.milvus.grpc.MutationResult;
import io.milvus.grpc.SearchResults;
import io.milvus.param.*;
import io.milvus.param.dml.InsertParam;
import io.milvus.param.dml.SearchParam;
import io.milvus.response.SearchResultsWrapper;
import org.jetbrains.annotations.NotNull;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.*;

@Service
public class MilvusVectorService {
    @Autowired
    private MilvusClient client;

    /**
     * 向 Milvus 插入单条向量（id 主键 + embedding 向量）
     */
    public void insert(Long id, List<Float> vector) {
        List<InsertParam.Field> fields = Arrays.asList(
                new InsertParam.Field("id", Collections.singletonList(id)),
                new InsertParam.Field("embedding", Collections.singletonList(vector))
        );
        InsertParam param = InsertParam.newBuilder()
                .withCollectionName("image_vectors")
                .withFields(fields)
                .build();
        R<MutationResult> res = client.insert(param);
        if (res.getStatus() != R.Status.Success.getCode()) {
            throw new RuntimeException("Milvus insert error: " + res.getMessage());
        }
    }

    public void insertAdditional(Long productId, List<Float> vector) {
        // 可用另一个集合或区分 type 字段
        List<InsertParam.Field> fields = Arrays.asList(
                new InsertParam.Field("product_id", Collections.singletonList(productId)),
                new InsertParam.Field("embedding", Collections.singletonList(vector)),
                new InsertParam.Field("type", Collections.singletonList("additional"))
        );
        InsertParam param = InsertParam.newBuilder()
                .withCollectionName("image_vectors")
                .withFields(fields)
                .build();
        client.insert(param);
    }

    /**
     * 根据特征向量检索最相似的TopN个商品ID
     * @param vector 查询图片的特征向量
     * @param topN   返回数量
     * @return List<Long> 相似商品ID列表，按相似度降序排列
     */
    public List<Long> searchTopN(List<Float> vector, int topN) {
        List<List<Float>> vectors = Collections.singletonList(vector);
        SearchParam searchParam = SearchParam.newBuilder()
                .withCollectionName("image_vectors")
                .withMetricType(MetricType.L2)
                .withOutFields(Collections.singletonList("id"))
                .withVectors(vectors)
                .withTopK(topN)
                .withVectorFieldName("embedding")
                .withParams("{\"nprobe\":16}") // 注意这里不是withParamsInJson
                .build();

        R<SearchResults> resp = client.search(searchParam);
        List<Long> idList = getLongs(resp);
        return idList;
    }

    @NotNull
    private static List<Long> getLongs(R<SearchResults> resp) {
        if (resp.getStatus() != R.Status.Success.getCode()) {
            throw new RuntimeException("Milvus search error: " + resp.getMessage());
        }
        SearchResultsWrapper wrapper = new SearchResultsWrapper(resp.getData().getResults());
        List<Long> idList = new ArrayList<>();
        for (int i = 0; i < wrapper.getRowRecords().size(); i++) {
            Object obj = wrapper.getFieldData("id", i);
            if (obj instanceof Long) {
                idList.add((Long) obj);
            } else if (obj instanceof Integer) { // 某些建表主键可能是 int
                idList.add(((Integer) obj).longValue());
            }
        }
        return idList;
    }
}

package app.image.service;

import io.milvus.client.MilvusServiceClient;
import io.milvus.grpc.DataType;
import io.milvus.param.MetricType;
import io.milvus.param.R;
import io.milvus.param.RpcStatus;
import io.milvus.param.collection.*;
import io.milvus.param.dml.InsertParam;
import io.milvus.param.dml.SearchParam;
import io.milvus.response.SearchResultsWrapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import jakarta.annotation.PostConstruct;
import java.util.*;

@Service
public class MilvusVectorService {
    private static final Logger log = LoggerFactory.getLogger(MilvusVectorService.class);
    private static final String COLLECTION = "image_vectors";
    private final MilvusServiceClient client;

    public MilvusVectorService(MilvusServiceClient client) {
        this.client = client;
    }

    @PostConstruct
    public void init() {
        // 1. 检查 Milvus 服务是否可用
        R<Boolean> hasResp = client.hasCollection(
                HasCollectionParam.newBuilder()
                        .withCollectionName(COLLECTION)
                        .build()
        );
        if (hasResp.getStatus() != R.Status.Success.getCode()) {
            log.error("无法连接到 Milvus 服务或检查 Collection 失败: code={}, msg={}",
                    hasResp.getStatus(), hasResp.getMessage());
            // 你可以选择抛异常终止启动，或 return 后在后续重试
            throw new IllegalStateException("Milvus 服务不可用: " + hasResp.getMessage());
        }

        // 2. 安全取出 Boolean，防止 null
        Boolean exists = hasResp.getData() != null && hasResp.getData();
        if (!exists) {
            log.info("Collection '{}' 不存在，开始创建…", COLLECTION);
            FieldType pk = FieldType.newBuilder()
                    .withName("id")
                    .withDataType(DataType.Int64)
                    .withPrimaryKey(true)
                    .withAutoID(false)
                    .build();
            FieldType vec = FieldType.newBuilder()
                    .withName("embedding")
                    .withDataType(DataType.FloatVector)
                    .withDimension(2048)
                    .build();

            R<RpcStatus> createResp = client.createCollection(
                    CreateCollectionParam.newBuilder()
                            .withCollectionName(COLLECTION)
                            .withFieldTypes(Arrays.asList(pk, vec))
                            .withShardsNum(1)
                            .build()
            );
            if (createResp.getStatus() != R.Status.Success.getCode()) {
                log.error("创建 Milvus Collection 失败: code={}, msg={}",
                        createResp.getStatus(), createResp.getMessage());
                throw new IllegalStateException("创建 Collection 失败: " + createResp.getMessage());
            }
            log.info("Collection '{}' 创建成功。", COLLECTION);
        } else {
            log.info("Collection '{}' 已存在，无需创建。", COLLECTION);
        }
    }


    public void insert(Long id, List<Float> vector) {

        List<InsertParam.Field> fields = Arrays.asList(
                // 主键字段
                new InsertParam.Field("id", Collections.singletonList(id)),
                // 向量字段（List<List<Float>>）
                new InsertParam.Field("embedding",
                        Collections.singletonList(vector))
        );
// 2. Build InsertParam，**没有** withPrimaryKeys(...) 方法
        InsertParam insertParam = InsertParam.newBuilder()
                .withCollectionName(COLLECTION)
                .withFields(fields)
                .build();
        client.insert(insertParam);
    }

    public List<Long> search(List<Float> queryVector, int topK) {
        SearchParam param = SearchParam.newBuilder()
                .withCollectionName(COLLECTION)
                .withMetricType(MetricType.IP)
                .withTopK(topK)
                .withVectors(Collections.singletonList(queryVector))
                .withVectorFieldName("embedding")
                .withParams("{\"nprobe\":10}")
                .build();
        R<io.milvus.grpc.SearchResults> resp = client.search(param);
        SearchResultsWrapper wrapper =
                new SearchResultsWrapper(resp.getData().getResults());
        List<SearchResultsWrapper.IDScore> list = wrapper.getIDScore(0);
        List<Long> ids = new ArrayList<>();
        list.forEach(s -> ids.add(s.getLongID()));
        return ids;
    }
}
package app.image.config;

import io.milvus.client.MilvusClient;
import io.milvus.client.MilvusServiceClient;
import io.milvus.grpc.DataType;
import io.milvus.param.ConnectParam;
import io.milvus.param.IndexType;
import io.milvus.param.MetricType;
import io.milvus.param.collection.*;
import io.milvus.param.index.CreateIndexParam;
import lombok.extern.slf4j.Slf4j;

import java.util.Arrays;

@Slf4j
public class MilvusCollectionCreator {

    public static void main(String[] args) {
        // 创建连接
        MilvusClient client = new MilvusServiceClient(
                ConnectParam.newBuilder()
                        .withHost("localhost")
                        .withPort(19530)
                        .build()
        );

        String collectionName = "product_vectors";

        // 如果已存在则先删除
        HasCollectionParam hasCollectionParam = HasCollectionParam.newBuilder()
                .withCollectionName(collectionName)
                .build();

        if (client.hasCollection(hasCollectionParam).getData()) {
            client.dropCollection(DropCollectionParam.newBuilder()
                    .withCollectionName(collectionName)
                    .build());
            log.info("旧 collection 已删除: {}", collectionName);
        }

        // 定义字段
        FieldType vectorId = FieldType.newBuilder()
                .withName("vector_id")
                .withDataType(DataType.VarChar)
                .withPrimaryKey(true)
                .withAutoID(false)
                .withMaxLength(128)
                .build();

        FieldType productId = FieldType.newBuilder()
                .withName("product_id")
                .withDataType(DataType.Int64)
                .build();

        FieldType imageType = FieldType.newBuilder()
                .withName("image_type")
                .withDataType(DataType.VarChar)
                .withMaxLength(32)
                .build();

        FieldType embedding = FieldType.newBuilder()
                .withName("embedding")
                .withDataType(DataType.FloatVector)
                .withDimension(864)
                .build();

        // 创建 Collection 参数
        CreateCollectionParam createParam = CreateCollectionParam.newBuilder()
                .withCollectionName(collectionName)
                .withDescription("Product image vectors")
                .withShardsNum(2)
                .addFieldType(vectorId)
                .addFieldType(productId)
                .addFieldType(imageType)
                .addFieldType(embedding)
                .build();

        client.createCollection(createParam);

        log.info("✅ Collection '{}' 创建成功", collectionName);

        // 可选：创建索引
        client.createIndex(CreateIndexParam.newBuilder()
                .withCollectionName(collectionName)
                .withFieldName("embedding")
                .withIndexName("embedding_idx")
                .withIndexType(IndexType.IVF_FLAT)
                .withMetricType(MetricType.L2)
                .withExtraParam("{\"nlist\":128}")
                .build());

        log.info("✅ embedding 字段索引创建成功");

        // 可选：加载进内存
        client.loadCollection(LoadCollectionParam.newBuilder()
                .withCollectionName(collectionName)
                .build());
        log.info("✅ Collection 已加载进内存");

        client.close();
    }
}

package app.image.service;

import app.image.service.MilvusVectorService;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Map;

@Slf4j
@Service
public class RedisTaskConsumer {

    @Autowired
    private RedisTemplate<String, Object> redis;

    @Autowired
    private ObjectMapper mapper;

    @Autowired
    private MilvusVectorService milvusService;

    /**
     * 定时消费 Redis 中的任务结果，将向量写入 Milvus
     */
//    @Scheduled(fixedDelay = 3000)
//    public void consumeAllTask() {
//        try {
//            // 找到所有带 :res 后缀的任务结果缓存key
//            var keys = redis.keys("task:*:res");
//            if (keys == null || keys.isEmpty()) {
//                return;
//            }
//            for (String key : keys) {
//                Object val = redis.opsForValue().get(key);
//                if (val == null) continue;
//
//                // 假设 val 是 JSON 转 Map，结构 {taskId, imageId, vector}
//                Map<?, ?> map = mapper.convertValue(val, Map.class);
//                if (map == null) continue;
//
//                Object imageIdObj = map.get("imageId");
//                Object vectorObj = map.get("vector");
//
//                if (imageIdObj == null || vectorObj == null) {
//                    log.warn("Task {} value missing fields", key);
//                    continue;
//                }
//
//                Long imageId = Long.valueOf(imageIdObj.toString());
//                List<Float> vector;
//
//                try {
//                    vector = (List<Float>) vectorObj;
//                } catch (ClassCastException e) {
//                    // 如果是 LinkedHashMap 转换的，需二次转换
//                    vector = ((List<?>) vectorObj).stream()
//                            .map(o -> Float.parseFloat(o.toString()))
//                            .toList();
//                }
//
//                // 插入 Milvus
//                milvusService.insert(imageId, vector);
//
//                // 处理完删除 Redis key
//                redis.delete(key);
//
//                log.info("Consumed and inserted vector for imageId: {}", imageId);
//            }
//        } catch (Exception e) {
//            log.error("RedisTaskConsumer consumeAllTask failed", e);
//        }
//    }
}

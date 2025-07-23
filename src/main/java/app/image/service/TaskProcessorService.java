package app.image.service;

import app.image.entity.TaskMessage;
import app.image.entity.ImageInfo;
import app.image.service.ImageEmbeddingService;
import app.image.service.ImageInfoService;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
@Slf4j
public class TaskProcessorService {
    private final RedisTemplate<String, String> redis;
    private final ObjectMapper mapper;
    private final ImageEmbeddingService embedService;
    private final MilvusVectorService milvusService;
    private final ImageInfoService infoService;

    public TaskProcessorService(RedisTemplate<String, String> redis,
                                ObjectMapper mapper,
                                ImageEmbeddingService embedService,
                                MilvusVectorService milvusService,
                                ImageInfoService infoService) {
        this.redis = redis;
        this.mapper = mapper;
        this.embedService = embedService;
        this.milvusService = milvusService;
        this.infoService = infoService;
    }

    @Scheduled(fixedDelay = 1000)
    public void process() {
        String msg = redis.opsForList().rightPop("taskQueue");
        if (msg == null) return;
        try {
            TaskMessage tm = mapper.readValue(msg, TaskMessage.class);
            ImageInfo info = infoService.getById(tm.getImageId());
            List<Float> vec = embedService.extract(info.getUrl());
            milvusService.insert(tm.getImageId(), vec);
            redis.opsForHash().put("taskStatus", tm.getTaskId(), "completed");
        } catch (Exception e) {
            log.error("Task processing failed", e);
        }
    }
}
package app.image.controller;

import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.web.bind.annotation.*;

import java.util.Collections;
import java.util.Map;

@RestController
@RequestMapping("/api/v1")
public class SearchController {
    private final RedisTemplate<String, String> redis;

    public SearchController(RedisTemplate<String, String> redis) {
        this.redis = redis;
    }

    @GetMapping("/search")
    public Map<String, Object> getResult(@RequestParam("taskId") String taskId) {
        String status = (String)redis.opsForHash().get("taskStatus", taskId);
        if (!"completed".equals(status)) {
            return Collections.singletonMap("status", status);
        }
        // demo 简化：这里只返回 completed，实际应返回结果列表
        return Collections.singletonMap("status", "completed");
    }
}
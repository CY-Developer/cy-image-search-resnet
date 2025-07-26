package app.image.controller;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/redis")
public class RedisController {

    @Autowired
    private RedisTemplate<String, Object> redis;

    @GetMapping("/task/{key}")
    public Object getTask(@PathVariable String key) {
        return redis.opsForValue().get(key);
    }
}

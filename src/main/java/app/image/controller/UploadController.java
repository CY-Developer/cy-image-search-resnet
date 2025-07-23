package app.image.controller;

import app.image.entity.TaskMessage;
import app.image.entity.UploadResponse;
import app.image.entity.ImageInfo;
import app.image.service.ImageInfoService;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.util.UUID;

@RestController
@RequestMapping("/api/v1")
public class UploadController {
    private final ImageInfoService infoService;
    private final RedisTemplate<String, String> redis;
    private final ObjectMapper mapper;

    public UploadController(ImageInfoService infoService,
                            RedisTemplate<String, String> redis,
                            ObjectMapper mapper) {
        this.infoService = infoService;
        this.redis = redis;
        this.mapper = mapper;
    }

    @PostMapping(value = "/upload", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public UploadResponse upload(@RequestParam("file") MultipartFile file,
                                 @RequestParam(value="sku", required=false) String sku) throws Exception {
        String fileName = UUID.randomUUID() + "_" + file.getOriginalFilename();
        String url = "/tmp/" + fileName; // 本地路径，生产替换OSS
        file.transferTo(new java.io.File(url));

        ImageInfo info = new ImageInfo();
        info.setSku(sku);
        info.setUrl(url);
        infoService.save(info);

        String taskId = UUID.randomUUID().toString();
        TaskMessage msg = new TaskMessage();
        msg.setTaskId(taskId);
        msg.setImageId(info.getId());
        redis.opsForList().leftPush("taskQueue", mapper.writeValueAsString(msg));
        redis.opsForHash().put("taskStatus", taskId, "processing");

        return new UploadResponse(taskId, "processing");
    }
}
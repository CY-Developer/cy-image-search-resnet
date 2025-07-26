package app.image.controller;

import app.image.service.MilvusVectorService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;
import java.util.List;

@RestController
@RequestMapping("/api/milvus")
public class MilvusController {

    @Autowired
    private MilvusVectorService milvusService;

    /**
     * 入库向量（简单示例，真实业务建议通过服务层调度任务，而不是暴露批量接口）
     */
    @PostMapping("/insert")
    public String insertVector(
            @RequestParam Long id,
            @RequestBody List<Float> vector
    ) {
        milvusService.insert(id, vector);
        return "ok";
    }
}

package app.image.entity;

import lombok.Data;

/**
 * 任务消息实体，表示一个待处理任务
 */
@Data
public class TaskMessage {
    private String taskId;    // 任务唯一ID
    private Long imageId;     // 这里 imageId 其实是 productId（商品ID）
}

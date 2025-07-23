package app.image.entity;

import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;

@Data
@TableName("images")
public class ImageInfo {
    @TableId
    private Long id;
    private String sku;
    private String url;
}
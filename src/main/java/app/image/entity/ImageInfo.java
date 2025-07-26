package app.image.entity;


import com.baomidou.mybatisplus.annotation.TableName;
import lombok.AllArgsConstructor;
import lombok.Data;

@Data
@TableName("your_image_info_table")  // 修改为实际表名
@AllArgsConstructor
public class ImageInfo {
    private Long id;
    private Long productId;
    private String url;   // 图片路径
    private String type;  // 图片类型
}
package app.image.entity;

import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;

@Data
@TableName("oc_product_image")
public class ProductImage {
    private Long productId;
    private String image; // 附图路径
    // 其他字段...
}

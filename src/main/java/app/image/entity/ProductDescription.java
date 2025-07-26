package app.image.entity;

import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;

@Data
@TableName("oc_product_description")
public class ProductDescription {
    private Long productId;
    private String description; // 详情富文本
}
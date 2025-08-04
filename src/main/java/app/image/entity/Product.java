package app.image.entity;
import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;

import java.util.Date;

@Data
@TableName("oc_product")
public class Product {
    @TableId
    private Long productId;
    private String image; // 主图路径
    private Date dateAdded;
    @TableField(exist = false)
    private String category;
    // 其他字段...
}



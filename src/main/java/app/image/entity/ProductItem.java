package app.image.entity;

import lombok.Getter;
import lombok.Setter;

@Setter
@Getter
public class ProductItem {
    private String productId;
    private String title;
    /** 数据库中的主图相对路径：如 catalog/2025/6/02/GUCCI 718154/画板 2.jpg */
    private String catalogPath;
    /** 后端可直接访问的图片 URL（由后端填充给前端用 <img>） */
    private String imageUrl;

    public ProductItem() {}
    public ProductItem(String productId, String title, String catalogPath, String imageUrl) {
        this.productId = productId;
        this.title = title;
        this.catalogPath = catalogPath;
        this.imageUrl = imageUrl;
    }

}
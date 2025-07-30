package app.image.entity;

import lombok.Data;
import java.util.List;

@Data
public class ProductImages {
    private Long productId;
    private String mainImage;
    private List<String> additionalImages;
    private List<String> detailImages;
    private String categoryName;
}

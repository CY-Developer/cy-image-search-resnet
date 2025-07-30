package app.image.entity;

import lombok.AllArgsConstructor;
import lombok.Data;

@Data
@AllArgsConstructor
public class ProductMatch {
    private Long productId;
    private String imageType;
    private float score;
}

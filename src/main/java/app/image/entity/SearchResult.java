package app.image.entity;

public class SearchResult {
    private final String productId;
    private final float score;

    public SearchResult(String productId, float score) {
        this.productId = productId;
        this.score = score;
    }

    public String getProductId() {
        return productId;
    }

    public float getScore() {
        return score;
    }
}

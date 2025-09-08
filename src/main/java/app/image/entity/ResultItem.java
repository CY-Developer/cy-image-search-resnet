package app.image.entity;

import lombok.Data;

@Data
public class ResultItem {
    public String productId;
    public double score;
    public String imagePath;
    public String state;

    @Override public String toString() {
        return "ResultItem{productId='" + productId + "', score=" + score +
                ", imagePath='" + imagePath + "', state='" + state + "'}";
    }
}
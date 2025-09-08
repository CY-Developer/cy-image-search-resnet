package app.image.service;

import app.image.entity.ResultItem;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import okhttp3.*;
import org.springframework.stereotype.Service;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

@Service
public class LocalImageSearchClient {
    private final OkHttpClient client = new OkHttpClient();
    private final ObjectMapper mapper = new ObjectMapper();


    /** 上传图片，返回Top-K的product_id列表（含可选信息） */
    public List<ResultItem> searchTopK(String imagePath, int k) throws IOException {
        File img = new File(imagePath);
        if (!img.exists()) {
            throw new IllegalArgumentException("Image not found: " + img.getAbsolutePath());
        }

        MediaType MEDIA_TYPE_JPG = MediaType.parse("image/jpeg");
        RequestBody fileBody = RequestBody.create(img, MEDIA_TYPE_JPG);

        MultipartBody requestBody = new MultipartBody.Builder()
                .setType(MultipartBody.FORM)
                .addFormDataPart("file", img.getName(), fileBody)
                .build();

        String baseUrl = "http://127.0.0.1:8000";
        HttpUrl url = HttpUrl.parse(baseUrl + "/search").newBuilder()
                .addQueryParameter("k", String.valueOf(k))
                .build();

        Request request = new Request.Builder()
                .url(url)
                .post(requestBody)
                .build();

        try (Response response = client.newCall(request).execute()) {
            if (!response.isSuccessful()) {
                throw new IOException("HTTP " + response.code() + ": " + response.message());
            }
            String body = response.body().string();
            JsonNode root = mapper.readTree(body);
            JsonNode arr = root.get("results");
            List<ResultItem> results = new ArrayList<>();
            if (arr != null && arr.isArray()) {
                for (JsonNode n : arr) {
                    ResultItem item = new ResultItem();
                    item.productId = n.get("product_id").asText();
                    item.score = n.get("score").asDouble();
                    item.imagePath = n.has("image_path") ? n.get("image_path").asText() : null;
                    item.state = n.has("state") ? n.get("state").asText() : null;
                    results.add(item);
                }
            }
            return results;
        }
    }

    /** 只要product_id的Top-10 */
    public List<String> searchTop10ProductIds(String imagePath) throws IOException {
        List<ResultItem> items = searchTopK(imagePath, 10);
        List<String> ids = new ArrayList<>();
        for (ResultItem it : items) ids.add(it.productId);
        return ids;
    }

    // 示例
    public static void main(String[] args) throws Exception {
        String img = "E:\\wrok\\app\\idea\\code\\cy-image-search-resnet\\model-project\\dummy_images\\WATCH\\4.OMEGA\\11.27\\原图\\3.OMEGA SEAMASTER BOUTIQUE EDITIONS WATCH 511.53.40.20.02.002\\主图\\511.53.40.20.02.002.jpg";
    }
}

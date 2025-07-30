package app.image.service;

import com.alibaba.fastjson.JSON;
import okhttp3.*;
import org.springframework.stereotype.Service;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.Map;

@Service
public class PythonVectorClient {

    private static final OkHttpClient client = new OkHttpClient();
    private static final String PYTHON_URL = "http://localhost:5000/extract";
    private static final String API_KEY = "93c1240be757f04a38c2aeb7e5cd7178";

    public static List<Float> extractVector(File imageFile) throws IOException {
        RequestBody fileBody = RequestBody.create(imageFile, MediaType.parse("image/jpeg"));
        MultipartBody requestBody = new MultipartBody.Builder()
                .setType(MultipartBody.FORM)
                .addFormDataPart("file", imageFile.getName(), fileBody)
                .build();

        Request request = new Request.Builder()
                .url(PYTHON_URL)
                .addHeader("X-API-Key", API_KEY)
                .post(requestBody)
                .build();

        try (Response response = client.newCall(request).execute()) {
            if (!response.isSuccessful()) throw new IOException("HTTP Error: " + response);
            String body = response.body().string();
            Map<String, Object> map = JSON.parseObject(body, Map.class);
            return (List<Float>) map.get("vector");
        }
    }
}

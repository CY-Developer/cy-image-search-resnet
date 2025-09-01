package app.image.service;

import com.alibaba.fastjson.JSON;
import okhttp3.*;
import org.springframework.stereotype.Service;

import java.io.File;
import java.io.IOException;
import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Map;

/**
 * Client for communicating with the Python vectorisation service.
 *
 * <p>
 * The original implementation only supported uploading a single image via
 * multipart form data to a fixed endpoint at port 5000.  This version
 * introduces category support and points to the improved service running on
 * port 8000.  Callers can optionally supply a category string which is
 * passed to the Python service as a query parameter.  If no category is
 * provided an empty string will be used.
 * </p>
 */
@Service
public class PythonVectorClient {

    private static final OkHttpClient client = new OkHttpClient();
    /**
     * Endpoint for single image extraction.  The category parameter will be
     * appended as a query string if present.
     */
    private static final String PYTHON_URL = "http://localhost:8000/extract";
    /**
     * API key for authenticating with the Python service.  Replace this
     * value with a secret known only to the server and client.
     */
    private static final String API_KEY = "93c1240be757f04a38c2aeb7e5cd7178";

    /**
     * Extract an embedding from the given image file.  This method is
     * retained for backwards compatibility.  It invokes the newer
     * {@link #extractVector(File, String)} with an empty category.
     *
     * @param imageFile the image file to embed
     * @return the embedding as a list of floats
     * @throws IOException if an HTTP or IO error occurs
     */
    public List<Float> extractVector(File imageFile) throws IOException {
        return extractVector(imageFile, "");
    }

    /**
     * Extract an embedding from the given image file with an optional
     * category hint.
     *
     * @param imageFile the image file to embed
     * @param category  an optional category name (e.g. "Shoes", "Bags").  May be null.
     * @return the embedding as a list of floats
     * @throws IOException if an HTTP or IO error occurs
     */
    @SuppressWarnings("unchecked")
    public List<Float> extractVector(File imageFile, String category) throws IOException {
        // Build multipart body with the image
        RequestBody fileBody = RequestBody.create(imageFile, MediaType.parse("image/jpeg"));
        MultipartBody requestBody = new MultipartBody.Builder()
                .setType(MultipartBody.FORM)
                .addFormDataPart("file", imageFile.getName(), fileBody)
                .build();
        // Append category as query parameter if provided
        String url = PYTHON_URL;
        if (category != null && !category.isEmpty()) {
            String encoded = URLEncoder.encode(category, StandardCharsets.UTF_8);
            url = url + "?category=" + encoded;
        }
        // Build the HTTP request
        Request request = new Request.Builder()
                .url(url)
                .addHeader("X-API-Key", API_KEY)
                .post(requestBody)
                .build();
        try (Response response = client.newCall(request).execute()) {
            if (!response.isSuccessful()) {
                throw new IOException("HTTP Error: " + response);
            }
            String body = response.body().string();
            Map<String, Object> map = JSON.parseObject(body, Map.class);
            Object vectorObj = map.get("vector");
            if (vectorObj instanceof List) {
                return (List<Float>) vectorObj;
            } else {
                throw new IOException("Unexpected response: missing vector field");
            }
        }
    }
}
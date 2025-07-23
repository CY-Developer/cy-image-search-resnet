package app.image.service;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;

import java.util.List;

@Service
public class ImageEmbeddingService {
    private final WebClient webClient;
    private final ObjectMapper mapper;

    public ImageEmbeddingService(@Value("${flask.url}") String flaskUrl,
                                 ObjectMapper mapper) {
        this.webClient = WebClient.builder().baseUrl(flaskUrl).build();
        this.mapper = mapper;
    }

    public List<Float> extract(String imageUrl) {
        JsonNode resp = webClient.post()
            .contentType(MediaType.APPLICATION_JSON)
            .bodyValue("{\"url\":\"" + imageUrl + "\"}")
            .retrieve()
            .bodyToMono(JsonNode.class)
            .block();
        return mapper.convertValue(resp.get("embedding"),
                                   mapper.getTypeFactory()
                                         .constructCollectionType(List.class, Float.class));
    }
}
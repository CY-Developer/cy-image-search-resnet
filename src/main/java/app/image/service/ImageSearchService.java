package app.image.service;

import io.milvus.client.MilvusClient;
import io.milvus.grpc.SearchResultData;
import io.milvus.grpc.SearchResults;
import io.milvus.param.MetricType;
import io.milvus.param.R;
import io.milvus.param.dml.SearchParam;
import io.milvus.response.SearchResultsWrapper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Service responsible for searching similar products using vectors.
 *
 * <p>
 * This implementation delegates the vector extraction to a Python
 * microservice via {@link PythonVectorClient}.  It then queries Milvus
 * for the nearest neighbours and aggregates the results at the
 * product level.  The default search parameters can be tuned via
 * {@code TOP_K_VECTOR} and {@code TOP_N_PRODUCT}.
 * </p>
 */
@Service
public class ImageSearchService {

    @Autowired
    private PythonVectorClient pVectorClient;

    @Autowired
    private MilvusClient milvusClient;

    private static final String COLLECTION_NAME = "product_vectors";
    private static final String VECTOR_FIELD = "embedding";
    /**
     * Number of nearest neighbours to request from Milvus.  Increasing this
     * value can improve recall at the cost of search latency.  In practice
     * values between 100 and 200 tend to give good trade‑offs for large
     * product catalogues.  Adjust according to your hardware and latency
     * requirements.
     */
    private static final int TOP_K_VECTOR = 150;
    /**
     * Number of products returned to the caller.  After aggregating
     * vectors by product ID, only the top N products with the highest
     * average similarity scores will be kept.  This should correspond
     * to how many results you wish to display to end users.
     */
    private static final int TOP_N_PRODUCT = 10;

    /**
     * Logger used for recording informational and diagnostic messages.
     */
    private static final Logger logger = LoggerFactory.getLogger(ImageSearchService.class);

    /**
     * Build the Milvus search parameter object.
     */
    private SearchParam buildSearchParam(String collectionName, String vectorField, List<?> queryVector, int topK) {
        List<Float> fixedVector = queryVector.stream()
                .map(v -> {
                    if (v instanceof Number) {
                        return ((Number) v).floatValue();
                    } else {
                        throw new IllegalArgumentException("向量必须是数字类型，实际类型: " + v.getClass().getName());
                    }
                })
                .collect(Collectors.toList());

        return SearchParam.newBuilder()
                .withCollectionName(collectionName)
                .withVectorFieldName(vectorField)
                .withOutFields(Collections.singletonList("product_id"))
                .withTopK(topK)
                // Use inner product to approximate cosine similarity; must match collection metric
                .withMetricType(MetricType.IP)
                .withVectors(Collections.singletonList(fixedVector))
                .withParams("{\"nprobe\":10}")
                .build();
    }

    /**
     * Aggregate raw search results by product ID and return the top N products
     * based on average score.
     */
    private List<Map<String, Object>> rankByProductId(SearchResultsWrapper wrapper, SearchResultData resultData, int topN) {
        List<?> productIds = wrapper.getFieldData("product_id", 0);
        List<?> scoreList = resultData.getScoresList();

        if (productIds == null || scoreList == null || productIds.size() != scoreList.size()) {
            throw new IllegalStateException("Milvus 返回数据维度不一致");
        }
        Map<String, List<Float>> grouped = new HashMap<>();
        for (int i = 0; i < productIds.size(); i++) {
            String pid = Objects.toString(productIds.get(i));
            grouped.computeIfAbsent(pid, k -> new ArrayList<>()).add(((Number) scoreList.get(i)).floatValue());
        }
        return grouped.entrySet().stream()
                .map(e -> {
                    Map<String, Object> map = new HashMap<>();
                    map.put("product_id", e.getKey());
                    map.put("score", average(e.getValue()));
                    return map;
                })
                .sorted((a, b) -> Float.compare((float) b.get("score"), (float) a.get("score")))
                .limit(topN)
                .collect(Collectors.toList());
    }

    private float average(List<Float> values) {
        return (float) values.stream().mapToDouble(Float::doubleValue).average().orElse(0.0);
    }

    
    /**
     * Perform a product search using the supplied image.  This is a convenience
     * overload that does not specify a category and therefore delegates to
     * {@link #searchProducts(MultipartFile, String)} with an empty
     * category.  Use this overload when the category of the uploaded
     * image is unknown or not important for cropping.
     *
     * @param multipartFile the uploaded image from the client
     * @return a ranked list of product IDs and their aggregated scores
     * @throws Exception if embedding fails or Milvus returns an error
     */
    public List<Map<String, Object>> searchProducts(MultipartFile multipartFile) throws Exception {
        return searchProducts(multipartFile, "");
    }

    /**
     * Perform a product search using the supplied image and optional category.
     *
     * <p>
     * The image is first written to a temporary file and then embedded via the
     * Python vectorisation service.  The resulting vector is used to query
     * Milvus.  The search results are aggregated by product ID and sorted
     * by descending average similarity.  Intermediate durations are logged
     * to aid in performance tuning.  Note that any temporary files will
     * be deleted when the method returns.
     * </p>
     *
     * @param multipartFile the uploaded image from the client
     * @param category      an optional category hint (e.g. "Shoes", "Bags").  If
     *                      empty no category specific cropping is applied
     * @return a list of maps containing product IDs and scores
     * @throws Exception if embedding fails or Milvus returns an error
     */
    public List<Map<String, Object>> searchProducts(MultipartFile multipartFile, String category) throws Exception {
        // Create a unique temporary file to hold the uploaded image.  Using
        // createTempFile ensures the file name does not collide with
        // concurrent requests.  The suffix is derived from the original
        // filename when possible.
        String suffix = ".jpg";
        try {
            String original = multipartFile.getOriginalFilename();
            if (original != null && original.lastIndexOf('.') != -1) {
                suffix = original.substring(original.lastIndexOf('.'));
            }
        } catch (Exception ignore) {
            // Default suffix already set
        }
        File imageFile = File.createTempFile("upload-", suffix);
        multipartFile.transferTo(imageFile);
        long startTotal = System.currentTimeMillis();
        logger.info("Starting search for file {} with category '{}'", imageFile.getName(), category);
        try {
            // Step 1: extract the embedding via the Python service
            long startExtract = System.currentTimeMillis();
            List<Float> queryVector = pVectorClient.extractVector(imageFile, category);
            long durationExtract = System.currentTimeMillis() - startExtract;
            logger.debug("Vector extraction took {} ms", durationExtract);
            // Step 2: build the search parameters and query Milvus
            SearchParam param = buildSearchParam(COLLECTION_NAME, VECTOR_FIELD, queryVector, TOP_K_VECTOR);
            long startSearch = System.currentTimeMillis();
            R<SearchResults> result = milvusClient.search(param);
            long durationSearch = System.currentTimeMillis() - startSearch;
            logger.debug("Milvus search took {} ms", durationSearch);
            if (result.getStatus() != R.Status.Success.getCode()) {
                throw new RuntimeException("Milvus 查询失败: " + result.getException().getMessage());
            }
            SearchResultData resultData = result.getData().getResults();
            SearchResultsWrapper wrapper = new SearchResultsWrapper(resultData);
            List<Map<String, Object>> ranked = rankByProductId(wrapper, resultData, TOP_N_PRODUCT);
            long totalDuration = System.currentTimeMillis() - startTotal;
            logger.info("Search completed in {} ms; returning {} products", totalDuration, ranked.size());
            return ranked;
        } finally {
            // Ensure the temporary file is removed from the filesystem
            if (!imageFile.delete()) {
                imageFile.deleteOnExit();
            }
        }
    }
}
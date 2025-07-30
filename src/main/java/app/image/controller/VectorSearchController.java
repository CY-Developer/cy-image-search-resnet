package app.image.controller;

import app.image.service.ImageSearchService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.util.*;

@RestController
@RequestMapping("/api/search")
public class VectorSearchController {

    @Autowired
    private ImageSearchService imageSearchService;

    @PostMapping
    public Map<String, Object> searchByImage(@RequestParam("file") MultipartFile file) {
        try {
            return imageSearchService.search(file);
        } catch (Exception e) {
            return Map.of("success", false, "message", e.getMessage());
        }
    }
}

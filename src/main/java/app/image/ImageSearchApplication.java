package app.image;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.scheduling.annotation.EnableScheduling;

@SpringBootApplication
@EnableScheduling  // 启用 @Scheduled
public class ImageSearchApplication {
    public static void main(String[] args) {
        SpringApplication.run(ImageSearchApplication.class, args);
    }
}
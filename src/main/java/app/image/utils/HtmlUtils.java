package app.image.utils;

import org.apache.commons.lang3.StringEscapeUtils;

import java.net.URLDecoder;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class HtmlUtils {
    private static final Pattern IMG_SRC = Pattern.compile("<img\\s+[^>]*src=\\\"(.*?)\\\"", Pattern.CASE_INSENSITIVE);

    /**
     * 从富文本中提取所有图片路径，并只保留 "/6/xxx/xxx.jpg" 这种格式
     */
    public static List<String> extractCatalogImagePaths(String html) {
        List<String> list = new ArrayList<>();
        if (html == null) return list;
        String unescaped = StringEscapeUtils.unescapeHtml4(html);
        Matcher matcher = IMG_SRC.matcher(unescaped);
        while (matcher.find()) {
            String raw = matcher.group(1);
            try { raw = URLDecoder.decode(raw, "UTF-8"); } catch (Exception ignored) {}
            int idx = raw.indexOf("/6/");
            if (idx >= 0) {
                String cut = raw.substring(idx); // 截取 /6/ 及其后面
                cut = cut.replaceAll("%20", " "); // 处理空格
                list.add(cut);
            }
        }
        return list;
    }
    /**
     * 只保留/6/及其后的路径，兼容各种前缀（如https、catalog、本地路径等）
     */
    public static String cutToSix(String path) {
        if (path == null) return null;
        int idx = path.indexOf("/6/");
        if (idx >= 0) {
            return path.substring(idx); // 从/6/及后面全部保留
        }
        return path; // 如果没有/6/则原样返回
    }
}

package app.image.mapper;

import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Select;

import java.util.List;
import java.util.Map;

@Mapper
public interface CategoryMapper {
    // 批量查询商品类目ID和类目名称
    @Select("<script>" +
            "SELECT ptc.product_id, cd.name AS category_name " +
            "FROM oc_product_to_category ptc " +
            "JOIN oc_category_description cd ON ptc.category_id = cd.category_id " +
            "WHERE ptc.product_id IN " +
            "<foreach item='item' collection='productIds' open='(' separator=',' close=')'>" +
            "#{item}" +
            "</foreach> " +
            "AND cd.language_id = 1 " + 
            "</script>")
    List<Map<String, Object>> getCategoriesByProductIds(List<Long> productIds);
}

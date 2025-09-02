package app.image.mapper;

import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Select;

import java.util.List;
import java.util.Map;

@Mapper
public interface CategoryMapper {
    // 批量查询商品类目ID和类目名称
//    @Select("<script>" +
//            "SELECT ptc.product_id, cd.name AS category_name " +
//            "FROM oc_product_to_category ptc " +
//            "JOIN oc_category_description cd ON ptc.category_id = cd.category_id " +
//            "JOIN oc_category oc ON ptc.category_id = oc.category_id " +
//            "WHERE ptc.product_id IN " +
//            "<foreach item='item' collection='productIds' open='(' separator=',' close=')'>" +
//            "#{item}" +
//            "</foreach> " +
//            "AND cd.language_id = 1  and  oc.parent_id  = 0 " +
//            "</script>")
    @Select("<script>" +
            "SELECT ptc.product_id, cd.name AS category_name\n" +
            "    FROM oc_product_to_category ptc\n" +
            "    JOIN oc_category_description cd ON ptc.category_id = cd.category_id\n" +
            "    JOIN oc_category oc ON ptc.category_id = oc.category_id\n" +
            "    WHERE ptc.product_id IN (select ptc.product_id from oc_product where YEAR(date_added) = '2025' AND MONTH(date_added) = '6')\n" +
            "                           AND cd.language_id = 1  and  oc.parent_id  = 0 and cd.name = 'Watches'" +
            "</script>")
    List<Map<String, Object>> getCategoriesByProductIds(List<Long> productIds);
}

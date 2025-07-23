package app.image.service;

import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import app.image.entity.ImageInfo;
import app.image.mapper.ImageInfoMapper;
import org.springframework.stereotype.Service;

@Service
public class ImageInfoService extends ServiceImpl<ImageInfoMapper, ImageInfo> {}
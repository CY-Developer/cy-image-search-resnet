from PIL import Image

def multi_scale_preprocess(image: Image.Image):
    return [image.convert("RGB")]

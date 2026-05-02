from PIL import Image
import io

def load_image(image_path_or_bytes):
    if isinstance(image_path_or_bytes, bytes):
        return Image.open(io.BytesIO(image_path_or_bytes))
    return Image.open(image_path_or_bytes)

def resize_image(image, size=(224, 224)):
    return image.resize(size)

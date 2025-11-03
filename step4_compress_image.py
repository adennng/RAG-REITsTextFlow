#压缩图片
import os
import math
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

def compress_image(input_path, max_pixels=12000000):
    """
    压缩图片，使总像素数不超过 max_pixels，并将压缩后的图片保存至原图片所在文件夹。
    返回压缩后图片的路径；如果图片无需压缩，则返回原始路径。
    """
    with Image.open(input_path) as img:
        width, height = img.size
        current_pixels = width * height
        print(f"原始图片尺寸: {width}x{height} = {current_pixels} 像素")
        
        if current_pixels <= max_pixels:
            print("图片像素数已在允许范围内，无需压缩。")
            return input_path  # 返回原始路径
        
        scale_factor = math.sqrt(max_pixels / current_pixels)
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        print(f"压缩比例: {scale_factor:.4f}，新的尺寸: {new_width}x{new_height}")
        
        resized_img = img.resize((new_width, new_height), Image.LANCZOS)
        folder, filename = os.path.split(input_path)
        name, ext = os.path.splitext(filename)
        output_filename = f"{name}_compressed{ext}"
        output_path = os.path.join(folder, output_filename)
        
        resized_img.save(output_path)
        print(f"压缩后的图片已保存至 {output_path}")
        return output_path

if __name__ == "__main__":
    input_path = r"***.png"
    compress_image(input_path)

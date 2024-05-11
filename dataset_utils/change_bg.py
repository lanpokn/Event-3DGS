import cv2
import os

def process_image(image):
    # 找到纯黑色的像素
    black_pixels_mask = (image[:, :, 0] == 0) & (image[:, :, 1] == 0) & (image[:, :, 2] == 0)
    # black_pixels_mask = (image[:, :, 0] == 166) & (image[:, :, 1] == 166) & (image[:, :, 2] == 166)
    # 将纯黑色的像素变为灰色
    image[black_pixels_mask] = [186, 186, 186]  # 灰色
    return image

def process_images_in_folder(input_folder, output_folder):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # 读取图像
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)

            # 处理图像
            processed_image = process_image(image)

            # 写入处理后的图像
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, processed_image)

            print(f"Processed image: {filename}")

# 调用函数处理图片文件夹
input_folder = "D:/2024/3DGS/dataset/nerf_synthetic/ship_colmap_easy/renders_old"
output_folder = "D:/2024/3DGS/dataset/nerf_synthetic/ship_colmap_easy/renders"
process_images_in_folder(input_folder, output_folder)
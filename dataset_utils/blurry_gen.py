import os
import cv2
import numpy as np
import shutil

def blurry_gen(source_folder, destination_folder):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    
    image_files = sorted([file for file in os.listdir(source_folder) if file.endswith('.png')])
    
    images = [cv2.imread(os.path.join(source_folder, file)) for file in image_files]
    num_images = len(images)
    
    # 定义5x5卷积核
    kernel = np.ones((7, 7), np.float32) / 49
    
    for i in range(1, num_images - 1):
        current_img = cv2.filter2D(images[i], -1, kernel)
        previous_img = cv2.filter2D(images[i - 1], -1, kernel)
        next_img = cv2.filter2D(images[i + 1], -1, kernel)
        
        blurred_img = (current_img.astype(np.float32) + previous_img.astype(np.float32) + next_img.astype(np.float32)) / 3
        blurred_img = np.clip(blurred_img, 0, 255).astype(np.uint8)
        
        output_path = os.path.join(destination_folder, image_files[i])
        cv2.imwrite(output_path, blurred_img)
    
    # 处理第一张图片（复制到目标文件夹）
    first_image_path = os.path.join(source_folder, image_files[0])
    first_image_output_path = os.path.join(destination_folder, image_files[0])
    shutil.copyfile(first_image_path, first_image_output_path)

    # 处理最后一张图片（复制到目标文件夹）
    last_image_path = os.path.join(source_folder, image_files[-1])
    last_image_output_path = os.path.join(destination_folder, image_files[-1])
    shutil.copyfile(last_image_path, last_image_output_path)

# 示例用法
source_folder = "D:/2024/3DGS/dataset/nerf_synthetic/lego_colmap_easy/renders"
destination_folder = "D:/2024/3DGS/dataset/nerf_synthetic/lego_colmap_easy/images_blurry"
blurry_gen(source_folder, destination_folder)

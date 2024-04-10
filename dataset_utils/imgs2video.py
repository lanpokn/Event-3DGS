import os
import cv2

def generate_video_from_images(folder_path, output_video_path, fps=25):
    # 获取文件夹中所有图片文件
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print("No image files found in the specified folder.")
        return
    
    # 读取第一张图片来获取尺寸信息
    first_image_path = os.path.join(folder_path, image_files[0])
    first_image = cv2.imread(first_image_path)
    height, width, _ = first_image.shape

    # 创建视频编写器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # 逐一读取图片并写入视频
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        image = cv2.imread(image_path)
        video_writer.write(image)

    # 释放资源
    video_writer.release()
    cv2.destroyAllWindows()

# 调用函数生成视频
generate_video_from_images("D:/2024/3DGS/dataset/nerf_synthetic/mic_colmap_easy/renders", "D:/2024/3DGS/dataset/nerf_synthetic/mic_colmap_easy/video.mp4", fps=25)
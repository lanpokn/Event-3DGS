import cv2
image_path = "D:/2024/3DGS/gaussian-splatting/output/myset/train/ours_7000/depth/00000.png"

# 读取图像
img = cv2.imread(image_path)
img = img/10
# 检查是否成功读取图像
if img is None:
    print(f"无法读取图像: {image_path}")

# 显示图像
cv2.imshow("Image Viewer", img)

# 等待用户按下任意键，然后关闭窗口
cv2.waitKey(0)
cv2.destroyAllWindows()
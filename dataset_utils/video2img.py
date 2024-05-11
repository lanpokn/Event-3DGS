import cv2
import os

def extract_frames(video_file, output_folder):
    # Open the video file
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Start frame counter
    frame_count = 0

    # Read frames and save as images
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Save frame as an image
        image_path = os.path.join(output_folder, f"{frame_count:05d}.jpg")
        cv2.imwrite(image_path, frame)

        frame_count += 1

    # Release video capture object
    cap.release()

# 调用函数生成视频
extract_frames("D:/2024/3DGS/dataset/e2vid_data/tunnel/tunnel.mp4","D:/2024/3DGS/dataset/e2vid_data/tunnel/renders")
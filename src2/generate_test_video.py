import cv2
import os

def create_test_video(image_folder, output_video_path, fps=30, img_size=(224, 224), duration_per_image=3):
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg") or img.endswith(".png")]
    images.sort()  # 按文件名排序

    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, img_size)

    # 每张图片显示的帧数
    frames_per_image = fps * duration_per_image

    for img_name in images:
        img_path = os.path.join(image_folder, img_name)
        img = cv2.imread(img_path)
        img_resized = cv2.resize(img, img_size)  # 调整图像尺寸

        # 将同一张图片写入多次以确保它显示指定的时间
        for _ in range(frames_per_image):
            out.write(img_resized)

    out.release()

# 设置文件夹路径和输出视频路径
image_folder = '/home/xu/Desktop/dl_demo/data/processed/cats_vs_dogs/test_video'  # 这里替换为存放猫狗图片的文件夹路径
output_video_path = '/home/xu/Desktop/dl_demo/data/cat_dog_test_video.avi'

create_test_video(image_folder, output_video_path)


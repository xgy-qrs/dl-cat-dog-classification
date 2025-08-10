import cv2
import torch
from model import initialize_model
from torchvision import transforms
import numpy as np

# 加载训练好的模型
def load_model(model_path, device):
    model = initialize_model(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# 图像预处理
def preprocess_image(frame, img_size):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(frame).unsqueeze(0)  # 增加batch维度

# 对视频进行推理
def infer_on_video(model, device, video_path, output_path=None):
    cap = cv2.VideoCapture(video_path)  # 读取本地视频文件
    if not cap.isOpened():
        print("Error: Unable to open video.")
        return
    
    # 获取视频信息
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    
    # 如果需要保存输出视频
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, frame_rate, (frame_width, frame_height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        input_tensor = preprocess_image(frame, img_size=224).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted = torch.max(outputs, 1)

        label = 'Cat' if predicted.item() == 0 else 'Dog'
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 显示推理结果
        cv2.imshow('Inference Video', frame)

        # 如果需要保存视频
        if output_path:
            out.write(frame)

        # 按 'q' 键退出
        if cv2.waitKey(int(frame_rate * 1)) & 0xFF == ord('q'):
            break

    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 设置设备（使用 GPU 如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载模型
    model = load_model('/home/xu/Desktop/dl_demo/old_results/epoch100/best_model.pth', device)  # 模型路径

    # 输入视频路径和输出视频路径（如果需要保存）
    input_video_path = '/home/xu/Desktop/dl_demo/data/cat_dog_test_video.avi'  # 本地视频路径 .avi / .mp4
    output_video_path = 'output_video.avi'  # 保存推理后的视频路径

    # 对视频进行推理并显示
    infer_on_video(model, device, input_video_path, output_video_path)


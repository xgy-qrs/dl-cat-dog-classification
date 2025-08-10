import cv2
import torch
from model import initialize_model  # 导入模型
from torchvision import transforms
import numpy as np

def load_model(model_path, device):
    model = initialize_model(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def preprocess_image(frame, img_size):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(frame).unsqueeze(0)  # 增加batch维度

def infer_on_video(model, device, video_source=0):
    cap = cv2.VideoCapture(video_source)  # 使用默认摄像头
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 预处理图像
        input_tensor = preprocess_image(frame, img_size=224).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted = torch.max(outputs, 1)

        # 显示推理结果
        label = 'Cat' if predicted.item() == 0 else 'Dog'
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Inference', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model('best_model.pth', device)  # 加载最好的模型
    infer_on_video(model, device)


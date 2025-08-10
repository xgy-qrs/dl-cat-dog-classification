"""
预测脚本：单张图像分类预测
作者：Xu Guangyuan
创作日期：2025/8/9
修改日期：2025/8/10 - 添加命令行参数支持
"""
import torch
from torchvision import transforms
from PIL import Image
from model import initialize_model
import os
import yaml
import sys
import argparse

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def predict_image(image_path, config):
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载模型
    model = initialize_model(device)
    model_path = os.path.join(config['model_load_dir'], config['model_name'])
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((config['img_size'], config['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 加载并预处理图像
    try:
        image = Image.open(image_path).convert('RGB')
    except FileNotFoundError:
        print(f"错误: 无法找到图像文件: {image_path}")
        print("请确保提供正确的图像路径")
        return None, None
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # 预测
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        _, predicted_class = torch.max(output, 1)
    
    # 解析结果
    class_names = ['Cat', 'Dog']
    predicted_label = class_names[predicted_class.item()]
    confidence = probabilities[predicted_class.item()].item()
    
    return predicted_label, confidence

if __name__ == "__main__":
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='猫狗图像分类预测')
    parser.add_argument('--image_path', type=str, required=True, 
                        help='要预测的图像路径')
    args = parser.parse_args()
    
    # 加载配置 - 使用绝对路径
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'configs', 'default.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)['prediction']
    
    # 获取项目根目录
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 将相对路径转换为绝对路径
    for key in ['model_load_dir']:
        config[key] = os.path.join(project_root, config[key])
    
    print(f"项目根目录: {project_root}")
    
    # 进行预测
    label, confidence = predict_image(args.image_path, config)
    
    if label and confidence:
        print(f"预测结果: {label} | 置信度: {confidence:.4f}")
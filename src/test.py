"""
测试脚本：模型评估与性能分析
作者：Xu Guangyuan
创作日期：2025/8/9
修改日期：2025/8/10 - 修复配置文件路径
"""
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import initialize_model
from utils import plot_confusion_matrix, generate_classification_report
import os
import yaml
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import sys
import os

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_model(config):
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 数据预处理
    test_transform = transforms.Compose([
        transforms.Resize((config['img_size'], config['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 加载数据集
    test_dataset = datasets.ImageFolder(
        config['val_dir'], transform=test_transform  # 使用验证集作为测试集
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4
    )
    
    # 加载模型
    model = initialize_model(device)
    model_path = os.path.join(config['model_load_dir'], config['model_name'])
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 测试评估
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    
    # 生成性能报告
    class_names = ['Cat', 'Dog']
    conf_matrix = confusion_matrix(all_labels, all_preds)
    cls_report = classification_report(all_labels, all_preds, target_names=class_names)
    
    # 保存结果
    plot_confusion_matrix(conf_matrix, class_names, config['results_dir'])
    generate_classification_report(cls_report, config['results_dir'])
    
    print("测试完成!")
    print("\n分类报告:")
    print(cls_report)

if __name__ == "__main__":
    # 加载配置 - 使用绝对路径
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'configs', 'default.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)['testing']
    
    # 获取项目根目录
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 将相对路径转换为绝对路径
    for key in ['val_dir', 'model_load_dir', 'results_dir']:
        config[key] = os.path.join(project_root, config[key])
    
    print(f"项目根目录: {project_root}")
    print(f"验证目录: {config['val_dir']}")
    print(f"模型加载目录: {config['model_load_dir']}")
    
    test_model(config)
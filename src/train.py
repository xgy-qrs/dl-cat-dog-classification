"""
训练脚本： 模型训练与验证
作者： Xu Guangyuan
修改日期： 2025/8/10 
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms
from model import initialize_model
from utils import save_model, plot_training_curves
import yaml
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import numpy as np
from PIL import Image
import random

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def train_model(config):
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 基础预处理（不包含数据增强）
    base_transform = transforms.Compose([
        transforms.Resize((config['img_size'], config['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 训练数据增强
    # train_transform = transforms.Compose([
    #     transforms.RandomResizedCrop(config['img_size']),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomRotation(15),
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # ])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(config['img_size'], scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.2),  # 添加垂直翻转
        transforms.RandomRotation(20),  # 增加旋转角度
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # 颜色抖动
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 随机平移
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    
    # 加载数据集
    train_dataset = datasets.ImageFolder(
        config['train_dir'], transform=train_transform
    )
    val_dataset = datasets.ImageFolder(
        config['val_dir'], transform=base_transform
    )
    
    # 创建随机索引
    train_indices = list(range(len(train_dataset)))
    val_indices = list(range(len(val_dataset)))
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        sampler=SubsetRandomSampler(train_indices),
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        sampler=SubsetRandomSampler(val_indices),
        num_workers=4
    )
    
    # 初始化模型
    model = initialize_model(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    # 添加学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.1, patience=5, verbose=True
    )

    # 在训练循环前添加
    best_acc = 0.0
    patience = 7  # 连续多少轮没有改进就停止
    no_improve = 0

    # 训练循环
    best_acc = 0.0
    train_losses = []
    val_losses = []
    accuracies = []
    
    for epoch in range(config['epochs']):
        # 训练阶段
        model.train()
        running_loss = 0.0
        
        # 监控批次类别分布
        batch_label_counts = {0: 0, 1: 0}
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']} [Train]"):
            images, labels = images.to(device), labels.to(device)
            
            # 更新批次类别计数
            for label in labels.cpu().numpy():
                batch_label_counts[label] += 1
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
        
        # 打印批次类别分布
        print(f"Epoch {epoch+1} 批次类别分布: 猫={batch_label_counts[0]}, 狗={batch_label_counts[1]}")
        
        # 计算平均损失
        epoch_loss = running_loss / len(train_loader.sampler)
        train_losses.append(epoch_loss)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{config['epochs']} [Val]"):
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader.sampler)
        val_losses.append(val_loss)
        accuracy = correct / total
        accuracies.append(accuracy)
        
        print(f"Epoch {epoch+1}/{config['epochs']} | "
              f"Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Accuracy: {accuracy:.4f}")
        
        # 在验证阶段后添加
        scheduler.step(accuracy)  # 根据验证准确率调整学习率

        # 保存最佳模型
        if accuracy > best_acc:
            best_acc = accuracy
            no_improve = 0
            save_model(model, os.path.join(config['model_save_dir'], 'best_model.pth'))
            print(f"保存最佳模型，准确率: {best_acc:.4f}")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"连续 {patience} 轮没有改进，停止训练")
                break
    
    # 保存最终模型
    save_model(model, os.path.join(config['model_save_dir'], 'final_model.pth'))
    
    # 绘制训练曲线
    plot_training_curves(train_losses, val_losses, accuracies, config['results_dir'])
    
    print("训练完成!")

if __name__ == "__main__":
    # 加载配置 - 使用绝对路径
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'configs', 'default.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)['training']
    
    # 获取项目根目录
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 将相对路径转换为绝对路径
    for key in ['train_dir', 'val_dir', 'model_save_dir', 'results_dir']:
        config[key] = os.path.join(project_root, config[key])
    
    print(f"项目根目录: {project_root}")
    print(f"训练目录: {config['train_dir']}")
    print(f"验证目录: {config['val_dir']}")
    
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    train_model(config)
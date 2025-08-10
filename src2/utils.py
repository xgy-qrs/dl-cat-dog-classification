"""
实用工具函数： 模型保存、结果可视化等
作者： Xu Guangyuan
创作日期： 2025/8/9
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from sklearn.metrics import confusion_matrix

def save_model(model, path):
    """保存模型到指定路径"""
    torch.save(model.state_dict(), path)
    print(f"模型已保存至: {path}")

# 修改 plot_training_curves 函数
def plot_training_curves(train_losses, val_losses, accuracies, save_dir):
    """绘制训练曲线并保存"""
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 5))
    
    # 损失曲线 (使用英文)
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # 准确率曲线 (使用英文)
    plt.subplot(1, 2, 2)
    plt.plot(accuracies, label='Accuracy', color='green')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'))
    plt.close()
    print(f"Training curves saved to: {os.path.join(save_dir, 'training_curves.png')}")

# 修改 plot_confusion_matrix 函数
def plot_confusion_matrix(cm, class_names, save_dir):
    """绘制混淆矩阵并保存"""
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.close()
    print(f"Confusion matrix saved to: {os.path.join(save_dir, 'confusion_matrix.png')}")

# 修改 generate_classification_report 函数
def generate_classification_report(report_str, save_dir):
    """保存分类报告文本"""
    os.makedirs(save_dir, exist_ok=True)
    report_path = os.path.join(save_dir, 'classification_report.txt')
    
    with open(report_path, 'w') as f:
        f.write(report_str)
    
    print(f"Classification report saved to: {report_path}")
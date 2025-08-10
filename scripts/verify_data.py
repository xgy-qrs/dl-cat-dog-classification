"""
数据验证脚本
作者：Xu Guangyuan
创作日期：2025/8/10
"""
import os
import random
from PIL import Image
import matplotlib.pyplot as plt

def verify_data(data_dir):
    # 获取所有类别
    classes = sorted(os.listdir(data_dir))
    print(f"找到类别: {classes}")
    
    # 为每个类别显示5个样本
    plt.figure(figsize=(15, 10))
    for i, cls in enumerate(classes):
        cls_dir = os.path.join(data_dir, cls)
        images = os.listdir(cls_dir)
        
        # 随机选择5张图片
        sample_images = random.sample(images, 5) if len(images) >= 5 else images
        
        for j, img_name in enumerate(sample_images):
            img_path = os.path.join(cls_dir, img_name)
            
            try:
                img = Image.open(img_path)
                plt.subplot(len(classes), 5, i*5 + j + 1)
                plt.imshow(img)
                plt.title(f"{cls} - {img_name}")
                plt.axis('off')
            except:
                print(f"无法打开图像: {img_path}")
    
    plt.tight_layout()
    plt.savefig("results_val/data_verification.png")
    print("数据验证图已保存至 results_val/data_verification.png")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='验证数据集')
    parser.add_argument('--data_dir', type=str, default='data/processed/cats_vs_dogs/train',
                        help='要验证的数据目录')
    args = parser.parse_args()
    
    verify_data(args.data_dir)

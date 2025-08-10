"""
模型定义： 实现猫狗分类的CNN网络
作者： Xu Guangyuan
创作日期： 2025/8/9
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# class CatDogCNN(nn.Module):
#     # 初始化函数，定义网络层结构
#     def __init__(self):
#         super(CatDogCNN, self).__init__()
#         self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#         self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.fc1 = nn.Linear(128 * 28 * 28, 512)  # 假设输入为224x224
#         self.fc2 = nn.Linear(512, 2)
#         self.dropout = nn.Dropout(0.5)

#     # 前向传播函数，定义数据在网络中的流动过程
#     def forward(self, x):
#         # 输入: (batch_size, 3, 224, 224)
#         x = self.pool(F.relu(self.conv1(x)))  # (batch_size, 32, 112, 112)
#         x = self.pool(F.relu(self.conv2(x)))  # (batch_size, 64, 56, 56)
#         x = self.pool(F.relu(self.conv3(x)))  # (batch_size, 128, 28, 28)
#         x = x.view(-1, 128 * 28 * 28)         # 将特征图展平为一维向量(batch_size, 128*28*28)，准备输入全连接层
#         x = self.dropout(x)
#         x = F.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = self.fc2(x)
#         return x

class CatDogCNN(nn.Module):
    def __init__(self):
        super(CatDogCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),  # 添加批归一化
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # 增加通道数
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 14 * 14, 512),  # 调整输入尺寸
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 2)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
        

def initialize_model(device):
    model = CatDogCNN()
    model = model.to(device)
    return model

if __name__ == "__main__":
    # 测试模型
    model = CatDogCNN()
    sample = torch.randn(1, 3, 224, 224)
    output = model(sample)
    print(f"模型输出形状: {output.shape}")
    print("模型结构测试通过!")
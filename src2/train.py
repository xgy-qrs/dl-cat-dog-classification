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
import random
from pathlib import Path
import numpy as np

class EarlyStopping:
    def __init__(self, patience=5, delta=0.001):
        self.patience = patience
        self.delta = delta
        self.best_loss = float('inf')
        self.counter = 0

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience

def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 数据预处理
    base_transform = transforms.Compose([
        transforms.Resize((config['img_size'], config['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(config['img_size'], scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(config['train_dir'], transform=train_transform)
    val_dataset = datasets.ImageFolder(config['val_dir'], transform=base_transform)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)

    model = initialize_model(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)

    best_acc = 0.0
    early_stopping = EarlyStopping(patience=5, delta=0.001)

    train_losses, val_losses, accuracies = [], [], []

    for epoch in range(config['epochs']):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']} [Train]"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)

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

        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        accuracy = correct / total
        accuracies.append(accuracy)

        print(f"Epoch {epoch+1}/{config['epochs']} | "
              f"Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Accuracy: {accuracy:.4f}")

        scheduler.step(accuracy)

        # Early stopping
        if early_stopping(val_loss):
            print(f"Early stopping at epoch {epoch+1}")
            break

        if accuracy > best_acc:
            best_acc = accuracy
            save_model(model, os.path.join(config['model_save_dir'], 'best_model.pth'))

    save_model(model, os.path.join(config['model_save_dir'], 'final_model.pth'))
    plot_training_curves(train_losses, val_losses, accuracies, config['results_dir'])
    print("训练完成!")

if __name__ == "__main__":
    config_path = Path(__file__).resolve().parent.parent / 'configs' / 'default.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)['training']

    project_root = Path(__file__).resolve().parent.parent
    for key in ['train_dir', 'val_dir', 'model_save_dir', 'results_dir']:
        config[key] = project_root / config[key]

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    train_model(config)


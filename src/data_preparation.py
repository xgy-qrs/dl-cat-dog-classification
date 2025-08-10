# 数据准备脚本：解压数据集、划分训练/验证集、预处理图像
# 作者：Xu Guangyuan
# 创作日期：2025/8/9
# 修改日期：2025/8/10 - 修复路径问题
import os  # 操作系统接口模块
import shutil  # 文件操作模块
import zipfile  # ZIP文件处理模块
from sklearn.model_selection import train_test_split  # 数据集划分工具
from torchvision import transforms  # 图像预处理模块
from PIL import Image  # 图像处理库
from tqdm import tqdm  # 进度条显示
import yaml  # YAML配置文件解析
import glob  # 文件路径匹配
from pathlib import Path  # 面向对象的路径操作
import random

def prepare_data(config):
    # 获取项目根目录（当前脚本的父目录的父目录）
    project_root = Path(__file__).resolve().parent.parent
    
    # 将配置中的相对路径转换为绝对路径
    for key in ['train_zip_path', 'test_zip_path', 'extract_dir', 
                'raw_cats_dir', 'raw_dogs_dir', 'train_dir', 
                'val_dir', 'test_dir']:
        config[key] = str(project_root / config[key])
    
    # 打印路径信息用于调试
    print(f"项目根目录: {project_root}")
    print(f"训练集路径: {config['train_zip_path']}")
    
    # 创建必要的目录结构（exist_ok=True表示目录存在时不报错）
    os.makedirs(config['raw_cats_dir'], exist_ok=True)
    os.makedirs(config['raw_dogs_dir'], exist_ok=True)
    os.makedirs(config['train_dir'], exist_ok=True)
    os.makedirs(config['val_dir'], exist_ok=True)
    os.makedirs(config['test_dir'], exist_ok=True)
    
    # 解压训练集（如果尚未解压）
    extract_dir = config['extract_dir']
    # 检查解压目录是否存在且非空
    if not os.path.exists(extract_dir) or len(os.listdir(extract_dir)) == 0:
        print("解压训练集...")
        with zipfile.ZipFile(config['train_zip_path'], 'r') as zip_ref:
            zip_ref.extractall(extract_dir)  # 解压整个ZIP文件
    else:
        print("训练集已解压，跳过解压步骤")
    
    # 检查解压后的文件结构
    extract_files = os.listdir(extract_dir)
    print(f"解压目录包含 {len(extract_files)} 个文件/目录")
    
    # 确定图片实际所在路径（处理可能的子目录结构）
    image_dir = extract_dir
    if 'train' in extract_files:  # 如果存在train子目录
        image_dir = os.path.join(extract_dir, 'train')
        print(f"检测到图片位于子目录: {image_dir}")
    else:
        print(f"图片位于根目录: {image_dir}")
    
    # 使用glob匹配猫狗图片文件（猫图片文件名以cat开头）
    print("分类猫狗图片...")
    cat_files = glob.glob(os.path.join(image_dir, 'cat*.jpg'))
    dog_files = glob.glob(os.path.join(image_dir, 'dog*.jpg'))  
    
    print(f"找到 {len(cat_files)} 张猫图片")
    print(f"找到 {len(dog_files)} 张狗图片")
    
    # 移动猫图片到原始目录
    for cat_file in tqdm(cat_files):  # tqdm显示进度条
        filename = os.path.basename(cat_file)
        dst_path = os.path.join(config['raw_cats_dir'], filename)
        shutil.move(cat_file, dst_path)  # 移动文件
    
    # 移动狗图片到原始目录
    for dog_file in tqdm(dog_files):
        filename = os.path.basename(dog_file)
        dst_path = os.path.join(config['raw_dogs_dir'], filename)
        shutil.move(dog_file, dst_path)
    
    # 划分训练集和验证集
    print("划分数据集...")
    # 获取移动后的图片列表（过滤jpg文件）
    cat_files = [f for f in os.listdir(config['raw_cats_dir']) if f.endswith('.jpg')]
    dog_files = [f for f in os.listdir(config['raw_dogs_dir']) if f.endswith('.jpg')]
    
    print(f"原始猫图片: {len(cat_files)}")
    print(f"原始狗图片: {len(dog_files)}")
    
    # 使用scikit-learn划分猫图片（固定随机种子确保可复现）
    train_cats, val_cats = train_test_split(
        cat_files, 
        test_size=config['val_ratio'],  # 验证集比例
        random_state=42  # 随机种子
    )
    # 划分狗图片
    train_dogs, val_dogs = train_test_split(
        dog_files, 
        test_size=config['val_ratio'], 
        random_state=42
    )
    
    # 打印划分结果
    print(f"训练集猫图片: {len(train_cats)}")
    print(f"验证集猫图片: {len(val_cats)}")
    print(f"训练集狗图片: {len(train_dogs)}")
    print(f"验证集狗图片: {len(val_dogs)}")
    
    # 定义图像预处理流程
    transform = transforms.Compose([
        transforms.Resize((config['img_size'], config['img_size'])),  # 调整尺寸
        transforms.ToTensor(),  # 转换为PyTorch张量
    ])
    
    # 处理并保存训练集图片
    print("处理训练集...")
    _process_images(train_cats, config['raw_cats_dir'], 
                   os.path.join(config['train_dir'], 'cats'), transform)
    _process_images(train_dogs, config['raw_dogs_dir'], 
                   os.path.join(config['train_dir'], 'dogs'), transform)
    
    # 处理并保存验证集图片
    print("处理验证集...")
    _process_images(val_cats, config['raw_cats_dir'], 
                   os.path.join(config['val_dir'], 'cats'), transform)
    _process_images(val_dogs, config['raw_dogs_dir'], 
                   os.path.join(config['val_dir'], 'dogs'), transform)
    
    # 处理测试集
    test_files = os.listdir(config['test_dir'])
    if len(test_files) == 0:  # 如果测试目录为空
        print("解压测试集...")
        with zipfile.ZipFile(config['test_zip_path'], 'r') as zip_ref:
            zip_ref.extractall(config['test_dir'])  # 解压测试集
    else:
        print("测试集已解压，跳过解压步骤")
        print(f"测试集包含 {len(test_files)} 张图片")

# 内部处理函数（实际执行图像预处理和保存）
def _process_images(file_list, src_dir, dst_dir, transform):
    os.makedirs(dst_dir, exist_ok=True)  # 确保目标目录存在
    for filename in tqdm(file_list):  # 带进度条遍历
        src_path = os.path.join(src_dir, filename)
        dst_path = os.path.join(dst_dir, filename)
        
        # 图像处理流程
        img = Image.open(src_path).convert('RGB')  # 打开并确保RGB格式
        img = transform(img)  # 应用预处理变换
        
        # 将张量转换回PIL图像并保存
        img = transforms.ToPILImage()(img)
        img.save(dst_path)  # 保存处理后的图片

# 主程序入口
if __name__ == "__main__":
    # 构建配置文件路径（位于上级目录的configs文件夹）
    config_path = os.path.join(os.path.dirname(__file__), '../configs/default.yaml')
    
    # 加载YAML配置文件
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)['data']  # 获取data部分的配置
    
    # 执行数据准备流程
    prepare_data(config)
    print("数据准备完成!")
import os
import shutil
import zipfile
from sklearn.model_selection import train_test_split
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import yaml
import glob
from pathlib import Path

def prepare_data(config):
    project_root = Path(__file__).resolve().parent.parent
    
    for key in ['train_zip_path', 'test_zip_path', 'extract_dir', 
                'raw_cats_dir', 'raw_dogs_dir', 'train_dir', 
                'val_dir', 'test_dir']:
        config[key] = str(project_root / config[key])

    os.makedirs(config['raw_cats_dir'], exist_ok=True)
    os.makedirs(config['raw_dogs_dir'], exist_ok=True)
    os.makedirs(config['train_dir'], exist_ok=True)
    os.makedirs(config['val_dir'], exist_ok=True)
    os.makedirs(config['test_dir'], exist_ok=True)

    extract_dir = config['extract_dir']
    if not os.path.exists(extract_dir) or len(os.listdir(extract_dir)) == 0:
        with zipfile.ZipFile(config['train_zip_path'], 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

    image_dir = extract_dir
    if 'train' in os.listdir(extract_dir):
        image_dir = os.path.join(extract_dir, 'train')

    cat_files = glob.glob(os.path.join(image_dir, 'cat*.jpg'))
    dog_files = glob.glob(os.path.join(image_dir, 'dog*.jpg'))

    for cat_file in tqdm(cat_files):
        filename = os.path.basename(cat_file)
        dst_path = os.path.join(config['raw_cats_dir'], filename)
        shutil.move(cat_file, dst_path)

    for dog_file in tqdm(dog_files):
        filename = os.path.basename(dog_file)
        dst_path = os.path.join(config['raw_dogs_dir'], filename)
        shutil.move(dog_file, dst_path)

    cat_files = [f for f in os.listdir(config['raw_cats_dir']) if f.endswith('.jpg')]
    dog_files = [f for f in os.listdir(config['raw_dogs_dir']) if f.endswith('.jpg')]

    train_cats, val_cats = train_test_split(cat_files, test_size=config['val_ratio'], random_state=42)
    train_dogs, val_dogs = train_test_split(dog_files, test_size=config['val_ratio'], random_state=42)

    transform = transforms.Compose([
        transforms.Resize((config['img_size'], config['img_size'])),
        transforms.ToTensor(),
    ])
    
    _process_images(train_cats, config['raw_cats_dir'], os.path.join(config['train_dir'], 'cats'), transform)
    _process_images(train_dogs, config['raw_dogs_dir'], os.path.join(config['train_dir'], 'dogs'), transform)
    _process_images(val_cats, config['raw_cats_dir'], os.path.join(config['val_dir'], 'cats'), transform)
    _process_images(val_dogs, config['raw_dogs_dir'], os.path.join(config['val_dir'], 'dogs'), transform)

def _process_images(file_list, src_dir, dst_dir, transform):
    os.makedirs(dst_dir, exist_ok=True)
    for filename in tqdm(file_list):
        src_path = os.path.join(src_dir, filename)
        dst_path = os.path.join(dst_dir, filename)
        
        img = Image.open(src_path).convert('RGB')
        img = transform(img)
        img = transforms.ToPILImage()(img)
        img.save(dst_path)

if __name__ == "__main__":
    config_path = Path(__file__).resolve().parent.parent / 'configs' / 'default.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)['data']
    
    prepare_data(config)
    print("数据准备完成!")


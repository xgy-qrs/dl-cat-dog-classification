# 猫狗识别深度学习项目

## 项目概述
本项目使用PyTorch实现了一个深度学习模型，用于识别图像中的猫和狗。包含完整的数据准备、模型训练、评估和部署流程。

## 项目结构
	dl_demo/
	├── data/
	│   ├── raw/                  # 原始数据集（不上传GitHub）
	│   │   ├── cats/             # 原始猫图片
	│   │   └── dogs/             # 原始狗图片
	│   └── processed/            # 处理后的数据集（不上传）
	│       └── cats_vs_dogs/     # 预处理后的数据集
	├── models/                   # 训练好的模型
	│   ├── best_model.pth        # 最佳模型权重
	│   └── final_model.pth       # 最终模型权重
	├── results/                  # 输出结果
	│   ├── training_curves.png   # 训练曲线
	│   ├── confusion_matrix.png  # 混淆矩阵
	│   ├── sample_images.png     # 样本图像
	│   └── classification_report.txt # 分类报告
	├── src/                      # 源代码
	│   ├── data_preparation.py   # 数据处理
	│   ├── model.py              # 模型定义
	│   ├── train.py              # 训练脚本
	│   ├── test.py               # 测试脚本
	│   ├── predict.py            # 预测脚本
	│   └── utils.py              # 辅助函数
	├── configs/                  # 配置文件
	│   └── default.yaml          # 默认配置
	├── scripts/                  # 实用脚本
	│   ├── download_data.sh      # 数据下载脚本
	│   └── setup_environment.sh  # 环境设置脚本
	├── requirements.txt          # Python依赖
	├── .gitignore                # Git忽略规则
	└── README.md                 # 项目文档

## 快速开始

### 1. 环境配置
```bash
# 创建conda虚拟环境
conda create -n catdog python=3.8 -y
conda activate catdog

# 安装依赖
pip install -r requirements.txt

# 或者运行设置脚本
bash scripts/setup_environment.sh

2. 准备数据

手动下载数据集:
	访问 百度网盘链接
	下载 dogs-vs-cats-redux-kernels-edition.zip
	解压并将 train.zip 和 test.zip 放入 data/raw/

运行数据准备脚本:
bash

python src/data_preparation.py

3. 训练模型
bash

python src/train.py

4. 评估模型
bash

python src/test.py

5. 进行预测
bash

python src/predict.py --image_path data/processed/cats_vs_dogs/test/unknown/test/*.jpg 

## 作者
Xu Guangyuan - Created on August 9, 2025

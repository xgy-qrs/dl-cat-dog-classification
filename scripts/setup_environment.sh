#!/bin/bash
# 环境设置脚本 - 使用conda虚拟环境
# 作者：Xu Guangyuan
# 创建日期：2025/8/9

echo "设置conda虚拟环境..."
conda create -n catdog python=3.8 -y
conda activate catdog

echo "安装依赖..."
pip install --upgrade pip
pip install -r ../requirements.txt

echo "环境设置完成!"
echo "请使用以下命令激活环境: conda activate catdog"
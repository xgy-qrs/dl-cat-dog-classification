#!/bin/bash
# 数据下载脚本（用户需手动下载）
# 作者：Xu Guangyuan
# 创建日期：2025/8/9
# 更新日期：2025/8/10
# 更新内容：添加目录创建命令

echo "准备数据目录..."
mkdir -p ../data/raw/cats
mkdir -p ../data/raw/dogs

echo "请手动下载数据集："
echo "1. 访问链接: https://pan.baidu.com/s/1qsvmq3uwqr79ykI5FblV8g"
echo "2. 下载 dogs-vs-cats-redux-kernels-edition.zip"
echo "3. 将下载文件放入 data/raw/ 目录"
echo ""
echo "下载完成后，请运行:"
echo "unzip ../data/raw/dogs-vs-cats-redux-kernels-edition.zip -d ../data/raw/"
echo "mv ../data/raw/train.zip ../data/raw/train.zip"
echo "mv ../data/raw/test.zip ../data/raw/test.zip"
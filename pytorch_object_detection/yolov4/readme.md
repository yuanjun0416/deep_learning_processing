## YOLOV4：You Only Look Once目标检测模型在pytorch当中的实现
---
本readme.md是我自己训练过程中总结的，本代码来自bubbliiiing/yolov4-pytorch,https://github.com/bubbliiiing/yolov4-pytorch

## 预训练权重下载
训练所使用的yolov4_weights.pth可在百度网盘中下载
链接：https://pan.baidu.com/s/1ygQlwjdwzAA8lIroe4UW2w 
提取码：xxxx

yolo4_weights.pth是coco数据集的权重，yolo4_voc_weights.pth是voc数据集的权重

## 训练步骤（这是我训练自己的数据集总结出来的）
### 训练自己的数据集
#### 一、预训练权重
1、如果使用预训练权重，请直接在上面百度网盘链接中下载预训练权重链接，我使用的是yolo4_weights.pth，也就是coco数据集的权重
2、将下载好的yolo4.weights放在model_data文件夹下
#### 二、数据集的准备


## YOLOV4：You Only Look Once目标检测模型在pytorch当中的实现
---
本readme.md是我自己训练过程中总结的，原代码为README_original.md(原先的也看一下，会有一些细节在其中)。本代码来自bubbliiiing/yolov4-pytorch,https://github.com/bubbliiiing/yolov4-pytorch

## 预训练权重下载
训练所使用的yolov4_weights.pth可在百度网盘中下载
链接：https://pan.baidu.com/s/1ygQlwjdwzAA8lIroe4UW2w 
提取码：xxxx

yolo4_weights.pth是coco数据集的权重，yolo4_voc_weights.pth是voc数据集的权重

## 训练步骤（这是我训练自己的数据集总结出来的）
### 训练自己的数据集
#### 一 训练权重
1. 如果使用预训练权重，请直接在上面百度网盘链接中下载预训练权重链接，我使用的是yolo4_weights.pth，也就是coco数据集的权重
2. 将下载好的yolo4.weights放在model_data文件夹下

#### 二 据集的准备
1. 本文使用的是VOC格式的数据集，将自己的VOC数据集按照VOCdevkit文件夹整理好
2. 注意一定要修该model_data/my_class.txt文件，将其修改成自己需要检测的类别
3. 使用voc_annotation.py生成需要用于训练的2007_train.txt、2007_val.txt文件，我将火灾数据集生成的2007_train.txt、2007_val.txt也已经放在yolov4中了
4. 如果VOCdevkit/ImageSets/Main中已经有分好的训练集和验证集，即train.txt、val.txt，需要生成跟这一样的数据集训练和验证的分类，那么需要将voc_annotation.py中参数修改为如下:
```python
annotation_mode = 2
class_path = 'model_data/my_classes.txt'
```

#### 三 开始网络训练
1. 训练的参数较多，均在train.py中，可以仔细查看
2. 必须修改的参数
```python
class_path = 'model_data/my_classes.txt'
```

## 预测步骤
### 得到best_model的map，直接调用get_map.py函数
1. 修改get_map.py中的参数如下
```python
map_mode = 4 # 这是使用pycocotools的，使用作者自己编写的会有报错，可能是有python库未安装
class_path = 'model_data/my_classes.txt'
```
2. 修改yolo.py中的参数如下
```python
"model_path" = 'logs/best_epoch_weights.pth'
```


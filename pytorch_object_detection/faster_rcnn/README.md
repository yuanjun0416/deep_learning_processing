# Faster R-CNN

## 该项目主要是来自pytorch官方torchvision模块中的源码
* https://github.com/pytorch/vision/tree/master/torchvision/models/detection

## 环境配置：
* Python3.6/3.7/3.8
* Pytorch1.7.1(注意：必须是1.6.0或以上，因为使用官方提供的混合精度训练1.6.0后才支持)
* pycocotools(Linux:`pip install pycocotools`; Windows:`pip install pycocotools-windows`(不需要额外安装vs))
* Ubuntu或Centos(不建议Windows)
* 最好使用GPU训练
* 详细环境配置见`requirements.txt`

## 文件结构：
```
  ├── backbone: 特征提取网络，可以根据自己的要求选择
  ├── network_files: Faster R-CNN网络（包括Fast R-CNN以及RPN等模块）
  ├── train_utils: 训练验证相关模块（包括cocotools）
  ├── my_dataset.py: 自定义dataset用于读取VOC数据集
  ├── train_mobilenet.py: 以MobileNetV2做为backbone进行训练
  ├── train_resnet50_fpn.py: 以resnet50+FPN做为backbone进行训练
  ├── train_multi_GPU.py: 针对使用多GPU的用户使用
  ├── predict.py: 简易的预测脚本，使用训练好的权重进行预测测试
  ├── validation.py: 利用训练好的权重验证/测试数据的COCO指标，并生成record_mAP.txt文件
  ├── split_data.py: 生成数据集中的Imagesets中的train.txt和val.txt文件
  └── pascal_voc_classes.json: pascal_voc标签文件
```

## 预训练权重下载地址（下载后放入backbone文件夹中）：
* MobileNetV2 backbone: https://download.pytorch.org/models/mobilenet_v2-b0353104.pth
* ResNet50+FPN backbone: https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth
* 注意，下载的预训练权重记得要重命名，比如在train_resnet50_fpn.py中读取的是`fasterrcnn_resnet50_fpn_coco.pth`文件，
  不是`fasterrcnn_resnet50_fpn_coco-258fb6c6.pth`
 
 
## 数据集，本例程使用的是PASCAL VOC2012数据集
* Pascal VOC2012 train/val数据集下载地址：http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
* 如果不了解数据集或者想使用自己的数据集进行训练，请参考我的bilibili：https://b23.tv/F1kSCK
* 使用ResNet50+FPN以及迁移学习在VOC2012数据集上得到的权重: 链接:https://pan.baidu.com/s/1ifilndFRtAV5RDZINSHj5w 提取码:dsz8

## 训练方法（如果直接训练pascal voc2012数据集，可以直接按照下列方法训练）
* 确保提前准备好数据集
* 确保提前下载好对应预训练模型权重
* 若要训练mobilenetv2+fasterrcnn，直接使用train_mobilenet.py训练脚本
* 若要训练resnet50+fpn+fasterrcnn，直接使用train_resnet50_fpn.py训练脚本
* 若要使用多GPU训练，使用`python -m torch.distributed.launch --nproc_per_node=8 --use_env train_multi_GPU.py`指令,`nproc_per_node`参数为使用GPU数量
* 如果想指定使用哪些GPU设备可在指令前加上`CUDA_VISIBLE_DEVICES=0,3`(例如我只要使用设备中的第1块和第4块GPU设备)
* `CUDA_VISIBLE_DEVICES=0,3 python -m torch.distributed.launch --nproc_per_node=2 --use_env train_multi_GPU.py`

## 训练自己的数据集的方法（按照下列方法来训练）
* 确保提前准备好数据集，如果数据集中只有下列的`JPEGImages`, `Annotations`文件，那么可以直接运行split_data.py文件产生`train.txt`，`val.txt`文件（split_data.py默认train：val = 1:1, 如要修改训练集和验证集的比例，可直接阅读split_data.py文件代码，只需要修改`eleven lines`的变量`val_rate`）
  ```
   ├── VOCtrainval
       ├── VOCdevkit
            ├── VOC2012
                 ├── JPEGImgaes
                 ├── Annotations
                 └── ImgaeSets
                      └── Main 
                           ├── train.txt
                           └── val.txt                                             
  ```
* 下载好对应的预训练模型权重（下载后放入backbone文件夹中）
* 若要训练mobilenetv2+fasterrcnn, 按照下列步骤来完成后，可直接运行`train_mobilenetv2.py`
  ```
  ├── ('fifty-eight lines'): 修改`voc_root = './VOCtrainval'`
  ├── (sixty-one lines): [如果需要使用混合精度训练(一定必须有GPU，当GPU显存不够时，可以使用)] 修改为`amp = True`
  ├── 注销掉`eighty-one lines and eighty-two lines` (如果您不使用Linux系统并行读书数据集的话，一般需要注销掉，反正我是注销掉的了)
      (`ninety lines and ninety-seven lines`): 修改`train_data_loader`中参数`num_workers = 0` 
      (`one hundred and seven lines`): 修改`val_data_loader`中参数`num_workers = 0`
  ├── (`one hundred and eleven lines`): 修改`create_model`中的`num_classes = 类别数+1`（这里填的是自己的目标检测识别的目标类别数 + 1（这个‘1’指的是背景））
  └── 修改pascal_voc_classes.json文件，将其中所写的pascal_voc检测的目标类别改为自己的数据集所要检测的类别
  ```
* 若要训练resnet50+fpn+fasterrcnn, 按照下列步骤来完成后，可直接运行`train_res50_fpn.py`
  ```
  ├── 注销掉`seventy-seven lines and seventy-eight lines` (如果您不使用Linux系统并行读书数据集的话，一般需要注销掉，反正我是注销掉的了)
      (`eighty-four lines and ninety-one lines`): 修改`train_data_loader`中参数`num_workers = 0`
      (`one hundred and one lines`): 修改`val_data_loader`中参数`num_workers = 0`
  ├── (`one hundred and nighty-three lines`): 修改数据集的根目录`defalut = './VOCtrainval'`
  ├── (`one hundred and nighty-five lines`): 修改检测目标的类别数`defalut = 类别数`（这里填的是自己的目标检测识别的目标类别数）
  ├── (`two hundred and twenty-one lines`): 使用GPU混合精度训练，修改`defalut = True` 
  └── 修改pascal_voc_classes.json文件，将其中所写的pascal_voc检测的目标类别改为自己的数据集所要检测的类别
  ```
* 如果要使用多GPU训练，本人没有尝试过，不太清楚

## 使用训练好了的权重去预测，按照下列步骤修改后，可以直接运行`predict.py`
  ```
  ├── 函数create_model中，总共有mobilenetv2和resnet50+fpn两种，只需要将不需要的那一种注销掉就可以，可以自行查看，一目了然
  ├── (fifty-two lines): 修改`create_model`中的`num_classes = 类别数+1`（这里填的是自己的目标检测识别的目标类别数 + 1（这个‘1’指的是背景））
  ├── (fifty-five lines): 将‘weights_path’改为训练好的权重参数路径
  └── (sixty-nine lines): 将‘original_img’改为需要预测的图片的路径
  ```
  
## 注意事项
* 在使用训练脚本时，注意要将`--data-path`(VOC_root)设置为自己存放`VOCdevkit`文件夹所在的**根目录**
* 由于带有FPN结构的Faster RCNN很吃显存，如果GPU的显存不够(如果batch_size小于8的话)建议在create_model函数中使用默认的norm_layer，
  即不传递norm_layer变量，默认去使用FrozenBatchNorm2d(即不会去更新参数的bn层),使用中发现效果也很好。
* 训练过程中保存的`results.txt`是每个epoch在验证集上的COCO指标，前12个值是COCO指标，后面两个值是训练平均损失以及学习率
* 在使用预测脚本时，要将`train_weights`设置为你自己生成的权重路径。
* 使用validation文件时，注意确保你的验证集或者测试集中必须包含每个类别的目标，并且使用时只需要修改`--num-classes`、`--data-path`和`--weights-path`即可，其他代码尽量不要改动

## 如果对Faster RCNN原理不是很理解可参考bilibili
* https://b23.tv/sXcBSP

## 进一步了解该项目，以及对Faster RCNN代码的分析可参考bilibili
* https://b23.tv/HvMiDy

## Faster RCNN框架图
![Faster R-CNN](fasterRCNN.png) 

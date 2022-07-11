import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module): 
    def __init__(self): #多继承需用到super函数，这是对继承父类的属性进行初始化
        super(LeNet, self).__init__() #self是习惯，大家都这么写，也可以写别的 
                                      #本人理解，后续理解错误，可能还会改，其实在这nn.Module是父类，继承的是LeNet类的父类（将父类的1初始化全部执行一遍）
                                      #其实效果跟 super().__init__()一样
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(in_features=32*5*5, out_features=120)  #将二维图片进行展平,(也可以直接使用reshape或者flatten(paddlepaddle深度学习框架中))
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)


    def forward(self, x):
        x = F.relu(self.conv1(x))    #input[3, 32, 32]->[rgb three channels， image size w, image size h]   output[16, 28, 28]【channels， charateristic pattern w, charateristic pattern h】
        x = self.pool1(x)            #input[16, 28, 28]   output[16, 14, 14] (same as above)     

        x = F.relu(self.conv2(x))    #input[16, 14, 14]   output[32, 10, 10]
        x = self.pool2(x)            #input[32, 10, 10]   output[32, 5, 5]

        x = x.view(-1, 32*5*5)       #output[batch_size, 32*5*5] 展平之后，展成一维向量
        x = F.relu(self.fc1(x))      #output[120]
        x = F.relu(self.fc2(x))      #output[84]
        x = self.fc3(x)              #output[10]

        return x

if __name__ == '__main__':
    input1 = torch.rand([32, 3, 32, 32])
    model = LeNet()
    print(model)
    output = model(input1)


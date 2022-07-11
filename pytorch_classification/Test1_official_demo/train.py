from pickletools import optimize
import torch
import torchvision
import torch.nn as nn
from model import LeNet
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

def train():
    transform = transforms.Compose([transforms.ToTensor(),     #[0, 255]->[0, 1] 具体可直接查看函数 
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  #见README问题一

    #50000张训练图片
    #第一次使用时需要将download设置为True才会自动下载数据集
    train_set = torchvision.datasets.CIFAR10(root='./data',
                                            train=True,
                                            download=False,
                                            transform=transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=36, shuffle=True, num_workers=0)

    #10000张图片
    #第一次使用时需要将download设置为True才会自动下载数据集
    test_set = torchvision.datasets.CIFAR10(root='./data',
                                            train=False,
                                            download=True,
                                            transform=transform)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=10000, shuffle=False, num_workers=0)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    test_loader_iter = iter(test_loader)  #创建一个迭代器对象
    test_image, test_label = test_loader_iter.next() 

    '''
    #when reading, change the batch_size in test_loader above from 10000 to 4 
    def imshow(img):
        img = img / 2 + 0.5     # unnormalize 反标准化处理
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0))) #在一开始载入图像是为[H, W, C], 在经过上面的代码transform.ToTensor处理后变为[C, H, W]，这里又变为[H, W, C]
        plt.show()

    # print labels
    print(' '.join(f'{classes[test_label[j]]:5s}' for j in range(4)))
    # show images
    imshow(torchvision.utils.make_grid(test_image))
    '''

    net = LeNet()
    loss_function = nn.CrossEntropyLoss()
    optimize = optim.Adam(net.parameters(), lr=0.001)

    for epoch in range(5): #loop over the dataset multiple
        
        running_loss = 0.0
        for step, data in enumerate(train_loader, start=0):
            #get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            #zero the parameter gradients
            optimize.zero_grad()  

            #forward + backward + optimize
            outputs = net(inputs)

            loss = loss_function(outputs, labels)
            loss.backward()
            optimize.step()

            #print statistics
            running_loss += loss.item()

            if step % 500 == 499: #print every 500 mini+batch
                with torch.no_grad():    #link:https://blog.csdn.net/weixin_43145941/article/details/114757673
                    outputs = net(test_image) #[batch, 10]
                    predict_y = torch.max(outputs, dim=1)[1]   #见README中的问题二                    
                    accuracy = torch.eq(predict_y, test_label).sum().item()/test_label.size(0) #见README中问题三

                    print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %
                            (epoch + 1, step + 1, running_loss / 500, accuracy))
                    running_loss = 0.0
    print('Finished Training')

    save_path = './Lenet.pth'
    torch.save(net.state_dict(), save_path)

if __name__ == '__main__':
    train()
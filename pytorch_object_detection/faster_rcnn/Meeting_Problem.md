# Meeting_Problem（本人初学）

### (1) results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))的含义 [train_mobilenetv2.py: forty-six lines]
   我们可以尝试使用以下代码解释一下
   ```
   import datetime
   print(datetime.datetime.now())
   results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
   print(results_file)
   ```
   output:
   ```
   2022-07-24 20:19:33.657781
   results20220724-201933.txt
   ```
   strftime参数介绍
   ```
   %Y 四位数的年份表示(0000-9999) %y 两位数的年份表示(00-99)
   %m 月份(01-12)                 %d 月内中的一天(01-31)
   %H 24小时制小时数(00-23)       %l 12小时制小时数(00-12)
   %M 分钟数(00-59)               %S 秒数(00-59)
   ```
   通过上面代码实验就显而易见了
   
   
### (2) train_sampler = torch.utils.data.RandomSampler(train_dataset) [train_mobilenetv2.py: seventy-seven.py]
   
   本人觉得这个就是随机采样数据集，是后面Dataloader中参数batch_size和shuffle的结合
   
   可以查看后几行代码，是一个对这的运用，可以好好感受一下
   
   链接也有代码介绍，可以感受一下：https://blog.csdn.net/lidc1004/article/details/115005612
   
### (3) model.parameters()的用法 [traini_mobilenetv2.py: one hundred and thirty-two lines]
   构建一个网络
   ```
   import torch.nn as nn
   #design Model
   class NeuralNetwork(nn.Module):
       def __init__(self):
           super(NeuralNetwork,self).__init__()
           self.flatten = nn.Flatten()
           self.linear_relu_stack = nn.Sequential(
               nn.Linear(2*2, 5),
               nn.ReLU(),
               nn.Linear(5,3),
               nn.ReLU()
           )

       def forward(self, x):
           x = self.flatten(x)
           logits = self.linear_relu_stack(x)
           return logits

   model = NeuralNetwork()
   print(model)
   print(model.parameters())  #迭代器可以使用for循环，即p for p in model.parameters() if p.requires_grad = True
   print(list(model.parameters())) 
   ```
   output:
   model:
   ```
   NeuralNetwork(
     (flatten): Flatten(start_dim=1, end_dim=-1)
     (linear_relu_stack): Sequential(
       (0): Linear(in_features=4, out_features=5, bias=True)
       (1): ReLU()
       (2): Linear(in_features=5, out_features=3, bias=True)
       (3): ReLU()
     )
   )
   ```
   model.parameters():
   ```
   <generator object Module.parameters at 0x000001E82C26DE40>
   ```
   list(model.parameters()):
   ```
   [Parameter containing:
   tensor([[-0.1933,  0.4739,  0.0775,  0.0639],
           [-0.4586, -0.3750, -0.2054, -0.0058],
           [ 0.4131, -0.0073, -0.2616, -0.4949],
           [-0.2943,  0.1258,  0.1990,  0.1450],
           [-0.0704, -0.0735,  0.4293, -0.2730]], requires_grad=True), Parameter containing:
   tensor([ 0.0749,  0.0115, -0.1905, -0.2829, -0.2272], requires_grad=True), 
   Parameter containing:
   tensor([[-0.2661,  0.2911, -0.3789, -0.4054, -0.3762],
           [ 0.1925,  0.3950,  0.2997, -0.3313,  0.0881],
           [-0.0518,  0.0329, -0.0005,  0.1160, -0.4402]], requires_grad=True), Parameter containing:
   tensor([-0.1595,  0.2987, -0.0165], requires_grad=True)]
   ```
### (4) f"{i:.4f}"和'  '.join(result_info)的用法 [train_mobilenetv2.py: one hundred and fifty-two lines]
   f"{i:.4f}"
   ```
   name = '张三'
   age = 22
   print(f"我的名字是{name}，今年{age:.4f}")
   ```
   output:
   ```
   我的名字是张三，今年22.0000
   ```
   '  '.join(result_info), 
   语法：'delimiter'.join(seq)
   
   delimiter: 分割符，可以为空
   seq: 要连接的元素序列、字符串、元组、字典
   ```
   result_info = "1234"
   print(','.join(result_info))
   result_info = ['1', '2', '3'] #不能是int类型的，如[1, 2, 3]
   print('|'.join(result_info))
   ```
   output:
   ```
   1,2,3,4
   1|2|3
   ```
   

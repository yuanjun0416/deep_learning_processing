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
   
   ```
   ```

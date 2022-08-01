# meeting problem

### (1) `img_path = os.path.join(dataset_dir.replace("labels", "images"),file_name.split(".")[0]) + ".jpg"`所遇到的一些小状况 [calculate_dataset.py: twenty-seven line]

result:
  ```
  ./my_yolo_dataset/train/images\2008_000008.jpg
  ```
The result can be directly recognized and run in the local IDE using VScode, 
but when running in an environment such as kaggle or goolge colab, an error will be reported

improve the code:
  ```
  img_path = dataset_dir.replace("labels", "images") + "/" + file_name.split(".")[0] + ".jpg" 
  ```
result:
  ```
  ./my_yolo_dataset/train/images/2008_000008.jpg
  ```

### (2) A brief introduction to how tensorboard are used [train.py: three hundered and one line]

* first: create SummaryWriter
  ```
  from tensorboardX import SummaryWriter
  #SummaryWriter压缩（包括）了所有内容
  writer = SummaryWriter('runs/exp-1')
  #创建writer object，log会被存入'runs/exp-1'
  writer2 = SummaryWriter()
  #用自动生成的文件名，文件夹类似'runs/Aug20-17-20-33'
  writer3 = SummaryWriter(comment='3x learning rate')
  #用自动生成的文件名创建writer3 object，注释（comment）会被加在文件名的后面。文件夹类似 'runs/Aug20-17-20-33-3xlearning rate'
  ```
  each subfolder is treated as a different experiment in tensorboard. Each time an experiment is run again with different setting, we need to modify the subfolder       names (such as run/exp2s, run/myexp, etc.)so that different experiments setting can be easily compared. Type tensorboard runs to compared differnet experiment in       tensorboard

* second: Add Scalar
  
  Scalar values are the easiest data type to work with. Usually, we save the loss value for each trainging step, or the correct rate for each epoch, and some times corresponding learning rate. The cost of saving scalar values is very low, just log whatever you think is important. A scalar value can be logged with the command `write.add_scalar('myscalar', values, iteration)`. It should be noted that it is not possible to input `a pytorch tensor` to the program. If x is a torch scalar tensor, remember to use `x.item()` to extract the scalar value.
  
### (3) `os.path.exists(path)` usage [build_utils/parse_config.py: seventy line]
  so far, `path=data/my_data.data` is equivalent to `path=./data/my_data.data`
  
### (4) `lambda` usage [trian.py: one hundred and thirty-eight line]

There are two ways to show it here, the second one is better to understand.

first: you can directly uncomment the following comments in the original code, and than set the breakpoint, the picture will be saved in 'LR.png'
  ```
  lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - hyp["lrf"]) + hyp["lrf"]  # cosine ## deteail usage: [meeting_problem.md-(4)]
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    scheduler.last_epoch = start_epoch  # 指定从哪个epoch开始

    # Plot lr schedule
    # y = []
    # for _ in range(epochs):
    #     scheduler.step()
    #     y.append(optimizer.param_groups[0]['lr'])
    # plt.plot(y, '.-', label='LambdaLR')
    # plt.xlabel('epoch')
    # plt.ylabel('LR')
    # plt.tight_layout()
    # plt.savefig('LR.png', dpi=300)

    # model.yolo_layers = model.module.yolo_layers
  ```
result:
![lambda_first_result](meeting_peoblem_images/lambda_first_result.png)

second: just run the following code directly
  ```
  import math
  import numpy as np
  import matplotlib.pyplot as plt


  lr0: 0.01  # initial learning rate (SGD=1E-2, Adam=1E-3)
  lrf: 0.2  # final OneCycleLR learning rate (lr0 * lrf)
  epochs=100
  lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - 0.2) + 0.2  # cosine

  list_lf=[]
  for i in range(epochs):
      value=lf(i)
      list_lf.append(value)

  x=np.linspace(0,102,100)
  plt.title('yolov5-s leaning rate line show ')
  plt.figure(1,figsize=(8,6))
  plt.plot(x,list_lf,color='blue',lw=2,linestyle='--')
  plt.show()
  ```
result:
![lambda_second_result](meeting_problem_images/lambda_second_result.png)

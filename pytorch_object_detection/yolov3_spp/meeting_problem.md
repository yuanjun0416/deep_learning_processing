# meeting problem

### (1) img_path = os.path.join(dataset_dir.replace("labels", "images"),file_name.split(".")[0]) + ".jpg"所遇到的一些小状况 [calculate_dataset.py: twenty-seven line]

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

first: create SummaryWriter
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

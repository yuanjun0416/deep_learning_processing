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
  
### (3) `os.path.exists(path)` usage
  so far, `path=data/my_data.data` is equivalent to `path=./data/my_data.data`
  

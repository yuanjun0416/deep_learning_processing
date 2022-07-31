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

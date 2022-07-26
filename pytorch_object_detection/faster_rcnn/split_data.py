import os
import random


def main():
    random.seed(0)  # 设置随机种子，保证随机结果可复现

    files_path = "./VOCdevkit/VOC2012/Annotations"       # 数据集根目录
    assert os.path.exists(files_path), "path: '{}' does not exist.".format(files_path)  # 判断该文件地址是否错误

    val_rate = 0.5  # 验证集占数据集的总比例

    files_name = sorted([file.split(".")[0] for file in os.listdir(files_path)])   # files_name: 以列表的形式存放文件的地址，如000000.xml->['000000',]
    files_num = len(files_name)
    val_index = random.sample(range(0, files_num), k=int(files_num*val_rate))      # 取数范围:(0, file_num), 取数个数: file_num*val_rate, val_index: 验证集
    train_files = []
    val_files = []
    for index, file_name in enumerate(files_name):  # 将files_name中的元素遍历一遍，并将其按照val_index的元素为索引存放于train_files和val_files文件中
        if index in val_index:
            val_files.append(file_name)
        else:
            train_files.append(file_name)

    try:
        train_f = open("train.txt", "x")    # train_f: 最终train.txt的文件地址， "train.txt": 文件存放地址(文件名), 'x': 新建一个文件，如果文件已存在，则报错
        eval_f = open("val.txt", "x")
        train_f.write("\n".join(train_files))   # 将train_files的列表内容写入train_f中
        eval_f.write("\n".join(val_files))
    except FileExistsError as e:
        print(e)
        exit(1)


if __name__ == '__main__':
    main()

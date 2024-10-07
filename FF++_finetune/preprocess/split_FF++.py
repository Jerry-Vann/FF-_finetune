import os
import random
import argparse

def split_dataset(data_dir, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2):
    # 获取所有视频文件
    all_files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]

    # 打乱文件列表
    random.shuffle(all_files)

    # 计算各个数据集的大小
    total_files = len(all_files)
    train_size = int(total_files * train_ratio)
    val_size = int(total_files * val_ratio)
    test_size = total_files - train_size - val_size

    # 划分数据集
    train_files = all_files[:train_size]
    val_files = all_files[train_size:train_size + val_size]
    test_files = all_files[train_size + val_size:]

    return train_files, val_files, test_files

def save_split(file_list, filename):
    with open(filename, 'w') as f:
        for file in file_list:
            f.write(f"{file}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset into train, val and test sets")
    parser.add_argument("data_dir", type=str, help="Path to the dataset directory")
    parser.add_argument("output_dir", type=str, help="Path to the output directory for saving the split files")
    args = parser.parse_args()

    train_files, val_files, test_files = split_dataset(args.data_dir)

    # 保存划分结果到 DFD 文件夹中
    save_split(train_files, os.path.join(args.output_dir, "train_FF++3500.txt"))
    save_split(val_files, os.path.join(args.output_dir, "val_FF++500.txt"))
    save_split(test_files, os.path.join(args.output_dir, "test_FF++1000.txt"))

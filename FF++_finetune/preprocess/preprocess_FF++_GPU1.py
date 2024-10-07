import os
import random
import argparse
import sys
import torch
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

# Set max_split_size_mb to avoid memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:22"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# 将当前工作目录添加到 Python 搜索路径中
sys.path.append(os.getcwd())
def crop_face(root: str):
    from util.face_sdk.face_crop import process_videos
    source_dirs = [
        # "manipulated_sequences/NeuralTextures" ,
        # "manipulated_sequences/FaceSwap",
        # "manipulated_sequences/Face2Face",
        # "manipulated_sequences/Deepfakes",
        "original_sequences/youtube"
    ]

    for source in source_dirs:
        source_dir = os.path.join(root, source)
        target_dir = os.path.join(root, "cropped_FF++", source)
        process_videos(source_dir, target_dir, ext="mp4")
        torch.cuda.empty_cache()  # 添加一行，手动释放显存

# 生成label.txt文件
def gen_label(root: str):
    test_txt_path = os.path.join(root, "label_FF++.txt")
    with open(test_txt_path, 'w') as f:
        # 遍历 manipulated_sequences 文件夹下的所有子文件夹，排除 DeepFakeDetection 和 FaceShifter 文件夹
        manipulated_sequences_dir = os.path.join(root, "manipulated_sequences")
        for subdir, _, files in tqdm(os.walk(manipulated_sequences_dir), desc='Processing manipulated_sequences'):
            if "DeepFakeDetection" in subdir or "FaceShifter" in subdir:
                continue
            for video in files:
                if video.endswith('.mp4'):  # 确保是视频文件
                    f.write(f"0 {video}\n")

        # 遍历 original_sequences 文件夹下的所有子文件夹，标记为 1
        original_sequences_dir = os.path.join(root, "original_sequences", "youtube")
        for subdir, _, files in tqdm(os.walk(original_sequences_dir), desc='Processing original_sequences'):
            for video in files:
                if video.endswith('.mp4'):  # 确保是视频文件
                    f.write(f"1 {video}\n")


# 生成test.txt文件
def gen_test(root: str):
    test_txt_path = os.path.join(root, "cropped_FF++", "test_FF++.txt")
    with open(test_txt_path, 'w') as f:
        # 遍历 manipulated_sequences 文件夹下的所有子文件夹，标记为 1
        manipulated_sequences_dir = os.path.join(root, "manipulated_sequences")
        for subdir, _, files in tqdm(os.walk(manipulated_sequences_dir), desc='Processing manipulated_sequences'):
            if "DeepFakeDetection" in subdir:  # 跳过 DeepFakeDetection 文件夹
                continue
            for video in files:
                if video.endswith('.mp4'):  # 确保是视频文件
                    parent_folder = os.path.basename(subdir)  # 获取上一级目录名称
                    f.write(f"{parent_folder}/{video} 1\n")

        # 遍历 original_sequences 文件夹下的所有子文件夹，标记为 0
        original_sequences_dir = os.path.join(root, "original_sequences")
        for subdir, _, files in tqdm(os.walk(original_sequences_dir), desc='Processing original_sequences'):
            if "actors" in subdir:  # 跳过 actors 文件夹
                continue
            for video in files:
                if video.endswith('.mp4'):  # 确保是视频文件
                    parent_folder = os.path.basename(subdir)  # 获取上一级目录名称
                    f.write(f"{parent_folder}/{video} 0\n")


# 创建命令行参数解析器
parser = argparse.ArgumentParser(description="Crop faces and generate test.txt file.")
# 添加一个参数，用于指定数据集的根目录
parser.add_argument('--data_dir', help='Root directory of the dataset')
# 解析命令行参数
args = parser.parse_args()

if __name__ == '__main__':

    # 获取数据集的根目录
    data_root = args.data_dir

    # 调用裁剪人脸的函数
    crop_face(data_root)

    # 生成test.txt文件
    # gen_test(data_root)

    # 生成label.txt文件
    # gen_label(data_root)
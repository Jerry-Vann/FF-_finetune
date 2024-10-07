import os
import random
import argparse
import sys
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")


# 将当前工作目录添加到 Python 搜索路径中
sys.path.append(os.getcwd())
def crop_face(root: str):
    from util.face_sdk.face_crop import process_videos
    source_dirs = [
        "manipulated_sequences/DeepFakeDetection", "original_sequences/actors"
    ]

    for source in source_dirs:
        source_dir = os.path.join(root, source)
        target_dir = os.path.join(root, "cropped", source)
        process_videos(source_dir, target_dir, ext="mp4")
# 生成label.txt文件
def gen_label(root: str):
    test_txt_path = os.path.join(root, "label_DFD.txt")
    with open(test_txt_path, 'w') as f:
        # 遍历 manipulated_sequences 文件夹下的所有子文件夹，标记为 0
        manipulated_sequences_dir = os.path.join(root, "manipulated_sequences", "DeepFakeDetection")
        for subdir, _, files in tqdm(os.walk(manipulated_sequences_dir), desc='Processing manipulated_sequences'):
            for video in files:
                if video.endswith('.mp4'):  # 确保是视频文件
                    f.write(f"0 {video}\n")

        # 遍历 original_sequences 文件夹下的所有子文件夹，标记为 1
        original_sequences_dir = os.path.join(root, "original_sequences", "actors")
        for subdir, _, files in tqdm(os.walk(original_sequences_dir), desc='Processing original_sequences'):
            for video in files:
                if video.endswith('.mp4'):  # 确保是视频文件
                    f.write(f"1 {video}\n")
#生成test.txt文件
def gen_test(root: str):
    test_txt_path = os.path.join(root, "test.txt")
    with open(test_txt_path, 'w') as f:
        # 遍历 DeepFakeDetection 文件夹下的所有子文件夹
        deepfake_detection_dir = os.path.join(root, "manipulated_sequences", "DeepFakeDetection")
        for subdir, _, files in tqdm(os.walk(deepfake_detection_dir), desc='Processing DeepFakeDetection'):
            for video in files:
                if video.endswith('.mp4'):  # 确保是视频文件
                    f.write(f"{video}\n")

        # 遍历 actors 文件夹下的所有子文件夹
        actors_dir = os.path.join(root, "original_sequences", "actors")
        for subdir, _, files in tqdm(os.walk(actors_dir), desc='Processing actors'):
            for video in files:
                if video.endswith('.mp4'):  # 确保是视频文件
                    f.write(f"{video}\n")

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
    # crop_face(data_root)

    # 生成label.txt文件
    gen_label(data_root)

    # 生成test.txt 文件
    # gen_test(data_root)
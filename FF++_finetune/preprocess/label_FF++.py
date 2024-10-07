import os
import argparse
from tqdm import tqdm

def gen_label(root: str):
    test_txt_path = os.path.join(root, "label_FF++.txt")
    with open(test_txt_path, 'w') as f:
        # 遍历 cropped_FF++/manipulated_sequences 文件夹下的所有子文件夹
        manipulated_sequences_dir = os.path.join(root, "cropped_FF++/manipulated_sequences")
        for subdir, _, files in tqdm(os.walk(manipulated_sequences_dir), desc='Processing manipulated_sequences'):
            if "DeepFakeDetection" in subdir or "FaceShifter" in subdir:
                continue
            for video in files:
                if video.endswith('.mp4'):  # 确保是视频文件
                    f.write(f"0 {video}\n")

        # 遍历 cropped_FF++/original_sequences 文件夹下的所有子文件夹，标记为 1
        original_sequences_dir = os.path.join(root, "cropped_FF++/original_sequences", "youtube")
        for subdir, _, files in tqdm(os.walk(original_sequences_dir), desc='Processing original_sequences'):
            for video in files:
                if video.endswith('.mp4'):  # 确保是视频文件
                    f.write(f"1 {video}\n")

# 创建命令行参数解析器
parser = argparse.ArgumentParser(description="Crop faces and generate test.txt file.")
# 添加一个参数，用于指定数据集的根目录
parser.add_argument('--data_dir', help='Root directory of the dataset')
# 解析命令行参数
args = parser.parse_args()

if __name__ == '__main__':

    # 获取数据集的根目录
    data_root = args.data_dir

    # 生成label.txt文件
    gen_label(data_root)
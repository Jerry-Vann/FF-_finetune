# 运行命令python FF++_extract.py --backbone marlin_vit_base_ytf --data_dir DFD

import argparse # 用于解析命令行参数
import os # 用于文件和目录操作
import sys # 提供对解释器使用或维护的变量的访问，以及与解释器强烈相关的函数
from pathlib import Path # 提供面向对象的文件系统路径操作

import numpy as np # 用于数值计算
import torch # 提供张量和自动微分支持，是PyTorch的核心库
from tqdm.auto import tqdm # 提供进度条功能，可以在循环中显示进度

from marlin_pytorch import Marlin # 导入Marlin模型
from marlin_pytorch.config import resolve_config # 导入配置解析函数

sys.path.append(".") # 将当前目录添加到Python的模块搜索路径中

if __name__ == '__main__':
    parser = argparse.ArgumentParser("CelebDF Feature Extraction") # 创建一个命令行参数解析器
    parser.add_argument("--backbone", type=str)# 添加一个字符串类型的命令行参数backbone
    parser.add_argument("--data_dir", type=str)# 添加一个字符串类型的命令行参数data_dir
    args = parser.parse_args()# 解析命令行参数

    model = Marlin.from_online(args.backbone)# 根据命令行参数创建Marlin模型
    config = resolve_config(args.backbone)# 解析模型的配置
    feat_dir = args.backbone + "_FF++"# 特征保存的目录名与backbone相同

    model.cuda()# 将模型移动到GPU
    model.eval()# 将模型移动到GPU

    raw_video_path = os.path.join(args.data_dir, "cropped_FF++")# 视频文件所在的目录
    all_videos = sorted(list(filter(lambda x: x.endswith(".mp4"), os.listdir(raw_video_path))))# 获取所有mp4格式的视频文件
    Path(os.path.join(args.data_dir, feat_dir)).mkdir(parents=True, exist_ok=True)# 创建特征保存的目录
    for video_name in tqdm(all_videos):# 遍历所有视频文件
        video_path = os.path.join(raw_video_path, video_name)# 视频文件的完整路径
        save_path = os.path.join(args.data_dir, feat_dir, video_name.replace(".mp4", ".npy"))# 特征保存的完整路径
        try:
            feat = model.extract_video(# 使用Marlin模型提取视频特征
                video_path, crop_face=False,# 不裁剪人脸
                sample_rate=config.tubelet_size, stride=config.n_frames,# 采样率和步长
                keep_seq=False, reduction="none")# 不保留序列和不做降维处理

        except Exception as e:# 捕获异常
            print(f"Video {video_path} error.", e)# 打印异常信息
            feat = torch.zeros(0, model.encoder.embed_dim, dtype=torch.float32)# 创建一个全零的特征张量
        np.save(save_path, feat.cpu().numpy())# 将特征保存为numpy格式文件

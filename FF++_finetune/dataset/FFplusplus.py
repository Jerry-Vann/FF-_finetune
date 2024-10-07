import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"   # 使用第一个GPU
from abc import ABC, abstractmethod
from itertools import islice
from typing import Optional

import ffmpeg
import numpy as np
import torch
import torchvision
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from marlin_pytorch.util import read_video, padding_video
from util.misc import sample_indexes, read_text, read_json

# 将 ffmpeg 安装路径添加到系统 PATH
# os.environ['PATH'] = "/home/wyj/bin:" + os.environ['PATH']
# 定义一个抽象基类CelebDF，继承自LightningDataModule和ABC
class FFplusplusBase(LightningDataModule, ABC):

    def __init__(self, data_root: str, label_file: str, split: str, task: str, data_ratio: float = 1.0, take_num: int = None):
        super().__init__()
        self.data_root = data_root# 数据集根目录
        self.label_file = label_file # 标签文件label.txt文件的路径(新添加)
        self.split = split# 数据集分割（如train, val, test）
        assert task in ("appearance", "action")# 任务类型只能是appearance或action
        self.task = task# 任务类型
        self.take_num = take_num# 任务类型
        # 读取文本文件并过滤空行
        self.name_list = list(
            filter(lambda x: x != "", read_text(os.path.join(data_root, f"{self.split}.txt")).split("\n")))
        #去除文件扩展名
        self.name_list = [os.path.splitext(name)[0] for name in self.name_list]
        # 使用函数读取label.txt文件
        self.video_names, self.labels = self.load_labels(self.label_file)# 获取视频名称和标签(新添加)
        self.metadata = {video_name: label for video_name, label in zip(self.video_names, self.labels)}# 读取元数据(新添加)
        # 根据data_ratio调整数据集大小
        if data_ratio < 1.0:
            self.name_list = self.name_list[:int(len(self.name_list) * data_ratio)]
        # 根据take_num调整数据集大小
        if take_num is not None:
            self.name_list = self.name_list[:self.take_num]
        # 打印数据集大小
        print(f"Dataset {self.split} has {len(self.name_list)} videos")
        # # 添加调试语句来检查元数据和视频名称列表
        # print("Sample metadata entries:", list(islice(self.metadata.items(), 5)))
        # print("Sample names from name_list:", self.name_list[:5])

    # label.txt的格式是 "label video_name" 的形式，每行一个样本
    def load_labels(self, label_file):
        # 初始化两个空列表，用于存储视频文件名和对应的标签
        video_names = []
        labels = []
        # 使用'with'语句打开标签文件，确保文件会被正确关闭
        # 'r'模式表示以只读方式打开文件
        with open(label_file, 'r') as file:
            for line in file:
                # 去除行首尾的空白字符，并按照空白字符分割行内容
                parts = line.strip().split()
                # 检查分割后的部分是否包含两个元素（一个标签和一个视频名）
                if len(parts) == 2:
                    # 解构赋值，将标签和视频名分别赋给label和video_name变量
                    label, video_name = parts
                    # 将标签转换为整数类型并添加到标签列表中
                    labels.append(int(label))
                    # 删除文件扩展名
                    video_name = os.path.splitext(video_name)[0]
                    # 将视频名添加到视频名列表中
                    video_names.append(video_name)
        return video_names, labels
    # 定义一个抽象方法，需要在子类中实现
    @abstractmethod
    def __getitem__(self, index: int):
        pass

    def __len__(self):
        return len(self.name_list)# 返回数据集大小

# for fine-tuning 处理 FFplusplus 数据集中的原始视频数据，根据剪辑帧数和时间采样率生成样本数据
class FFplusplus(FFplusplusBase):

    def __init__(self,
        root_dir: str, # 根目录
        label_file: str, #标签目录
        split: str, # 拆分方式
        task: str, # 任务类型
        clip_frames: int, # 剪辑帧数
        temporal_sample_rate: int, # 时间采样率
        data_ratio: float = 1.0, # 数据比例
        take_num: Optional[int] = None # 取样数量
    ):
        super().__init__(root_dir, label_file, split, task, data_ratio, take_num)
        self.clip_frames = clip_frames
        self.temporal_sample_rate = temporal_sample_rate
    # 根据给定的索引获取数据集中指定位置的样本数据
    def __getitem__(self, index: int):
        # 获取标签和视频路径：从元数据中获取指定索引样本的标签，并构建相应的视频文件路径
        y = self.metadata.get(self.name_list[index], 0)# 获取标签
        video_path = os.path.join(self.data_root, "cropped_FF++", self.name_list[index] + ".mp4")
        # 使用 FFmpeg 获取视频帧数：使用 FFmpeg 库的 probe() 方法获取视频文件的帧数
        probe = ffmpeg.probe(video_path)["streams"][0]
        n_frames = int(probe["nb_frames"])
        #处理视频长度不足的情况：如果视频帧数不足以满足剪辑帧数，则从头开始读取视频，并进行填充处理，确保视频长度与指定的剪辑帧数相同
        if n_frames <= self.clip_frames:
            video = read_video(video_path, channel_first=True).video / 255
            # pad frames to 16 将帧设置为16帧
            video = padding_video(video, self.clip_frames, "same")  # (T, C, H, W)
            video = video.permute(1, 0, 2, 3)  # (C, T, H, W)
            return video, torch.tensor(y, dtype=torch.long)
        elif n_frames <= self.clip_frames * self.temporal_sample_rate:
            # reset a lower temporal sample rate 重置较低的时间采样率
            sample_rate = n_frames // self.clip_frames
        else:
            sample_rate = self.temporal_sample_rate
        # sample frames
        video_indexes = sample_indexes(n_frames, self.clip_frames, sample_rate)
        reader = torchvision.io.VideoReader(video_path)
        fps = reader.get_metadata()["video"]["fps"][0]
        reader.seek(video_indexes[0].item() / fps, True)
        frames = []
        for frame in islice(reader, 0, self.clip_frames * sample_rate, sample_rate):
            frames.append(frame["data"])
        video = torch.stack(frames) / 255  # (T, C, H, W)
        video = video.permute(1, 0, 2, 3)  # (C, T, H, W)
        assert video.shape[1] == self.clip_frames, video_path
        return video, torch.tensor(y, dtype=torch.long).bool()


# For linear probing
class FFplusplusFeatures(FFplusplusBase):

    def __init__(self, root_dir: str,
        feature_dir: str,
        label_file: str,
        split: str,
        task: str,
        temporal_reduction: str,
        data_ratio: float = 1.0,
        take_num: Optional[int] = None,
        batch_size: int = 1  # 添加 batch_size 参数
    ):
        super().__init__(root_dir, label_file, split, task, data_ratio, take_num)
        self.feature_dir = feature_dir# 特征目录
        self.temporal_reduction = temporal_reduction# 时间降维方式
        self.batch_size = batch_size  # 添加 batch_size 属性

    def __getitem__(self, index: int, batch_size: Optional[int] = None):
        batch_size = batch_size if batch_size is not None else self.batch_size  # 修改此处
        # 根据索引值拼接特征文件的完整路径
        feat_path = os.path.join(self.data_root, self.feature_dir, self.name_list[index] + ".npy")

        x = torch.from_numpy(np.load(feat_path)).float()# 加载特征

        if x.size(0) == 0:
            x = torch.zeros(1, 768, dtype=torch.float32)# 如果特征为空，则填充零
        # 根据时间降维方式处理特征
        if self.temporal_reduction == "mean":
            x = x.mean(dim=0)
        elif self.temporal_reduction == "max":
            x = x.max(dim=0)[0]
        elif self.temporal_reduction == "min":
            x = x.min(dim=0)[0]
        else:
            raise ValueError(self.temporal_reduction)

        y = self.metadata.get(self.name_list[index], 0)# 获取标签
        # y = self.metadata["labels"][self.name_list[index]]["attributes"][self.task]

        # Ensure the size of y matches the batch size
        y = torch.tensor([y] * batch_size, dtype=torch.long).bool()
        # print(f"Video: {self.name_list[index]}, Label: {y}") #打印调试语句
        # # 添加调试语句检查标签
        # print(f"Index: {index}, Video: {self.name_list[index]}, Label: {y}")
        # 确保y的大小与批大小匹配
        y = torch.tensor([y], dtype=torch.long).repeat(batch_size).bool()
        return x, y# 返回特征和标签

# CelebvHqDataModule类是一个PyTorch Lightning数据模块，用于加载数据
class FFplusplusDataModule(LightningDataModule):

    def __init__(self, root_dir: str,
        load_raw: bool,
        task: str,
        label_file: str,
        batch_size: int,
        num_workers: int = 0,
        clip_frames: int = None,
        temporal_sample_rate: int = None,
        feature_dir: str = None,
        temporal_reduction: str = "mean",
        data_ratio: float = 1.0,
        take_train: Optional[int] = None,
        take_val: Optional[int] = None,
        take_test: Optional[int] = None
    ):
        super().__init__()

        self.root_dir = root_dir# 数据根目录
        self.task = task# 任务类型
        self.label_file = label_file # 标签文件
        self.batch_size = batch_size# 批大小
        self.num_workers = num_workers# 工作进程数
        self.clip_frames = clip_frames# 帧数
        self.temporal_sample_rate = temporal_sample_rate# 时间采样率
        self.feature_dir = feature_dir# 特征目录
        self.temporal_reduction = temporal_reduction# 时间降维
        self.load_raw = load_raw# 是否加载原始数据
        self.data_ratio = data_ratio# 数据比例
        self.take_train = take_train# 训练集大小
        self.take_val = take_val# 验证集大小
        self.take_test = take_test # 测试集大小

        if load_raw:
            assert clip_frames is not None
            assert temporal_sample_rate is not None
        else:
            assert feature_dir is not None
            assert temporal_reduction is not None

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None):
        if self.load_raw:
            self.train_dataset = FFplusplus(self.root_dir, self.label_file,"train_FF++3500", self.task, self.clip_frames,
                self.temporal_sample_rate, self.data_ratio, self.take_train)
            self.val_dataset = FFplusplus(self.root_dir, self.label_file,"val_FF++750", self.task, self.clip_frames,
                self.temporal_sample_rate, self.data_ratio, self.take_val)
            self.test_dataset = FFplusplus(self.root_dir, self.label_file, "test_FF++750", self.task, self.clip_frames,
                self.temporal_sample_rate, 1.0, self.take_test)
        else:
            self.train_dataset = FFplusplusFeatures(self.root_dir, self.feature_dir, self.label_file, "train_FF++3500", self.task,
                self.temporal_reduction, self.data_ratio, self.take_train)
            self.val_dataset = FFplusplusFeatures(self.root_dir, self.feature_dir, self.label_file, "val_FF++500", self.task,
                self.temporal_reduction, self.data_ratio, self.take_val)
            self.test_dataset = FFplusplusFeatures(self.root_dir, self.feature_dir, self.label_file, "test_FF++1000", self.task,
                self.temporal_reduction, 1.0, self.take_test)


    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
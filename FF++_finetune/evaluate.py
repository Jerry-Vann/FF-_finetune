import argparse

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torchmetrics import AUROC
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter

from dataset.DFD import DFDDataModule
from marlin_pytorch.config import resolve_config
from marlin_pytorch.util import read_yaml
from model.classifier import Classifier
from util.earlystop_lr import EarlyStoppingLR
from util.lr_logger import LrLogger
from util.seed import Seed
from util.system_stats_logger import SystemStatsLogger
from torchmetrics.classification import BinaryAUROC

def evaluate_DFD(args, config):
    data_path = args.data_path
    resume_ckpt = args.resume
    n_gpus = args.n_gpus
    max_epochs = args.epochs
    label_file = args.label_file


    model = Classifier.load_from_checkpoint("checkpoint/MARLIN_FF++.ckpt")
    accelerator = "cpu" if args.n_gpus == 0 else "gpu"
    trainer = Trainer(log_every_n_steps=1, devices=1 if args.n_gpus > 0 else 0, accelerator=accelerator, benchmark=True,
        logger=False, enable_checkpointing=False)
    Seed.set(42)
    model.eval()

    finetune = config["finetune"]
    learning_rate = config["learning_rate"]
    task = config["task"]

    dm = DFDDataModule(
        data_path, finetune, task, label_file,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        feature_dir=config["backbone"],
        temporal_reduction=config["temporal_reduction"]
    )

    # 设置数据模块
    dm.setup()  # 确保在使用数据加载器之前调用setup()方法

    # collect predictions收集预测结果
    preds = trainer.predict(model, dm.test_dataloader())
    preds = torch.cat(preds).view(-1)  # 展平张量
    # preds = torch.cat(preds)

    # collect ground truth收集真实标签
    ys = torch.tensor([], dtype=torch.bool)  # 初始化 ys 为空张量

    for i, (_, y) in enumerate(tqdm(dm.test_dataloader())):
        ys = torch.cat([ys, y.view(-1)])  # 展平并收集真实标签

    preds = preds.sigmoid()
    acc = ((preds > 0.5) == ys).float().mean()
    auc = model.auc_fn(preds, ys)

    results = {
        "acc": acc,
        "auc": auc
    }

    print(results)

    # # 将预测结果保存到 txt 文件（修改3）
    # with open("predictions.txt", "w") as f:
    #     f.write("Predictions\tGround Truth\n")
    #     for pred, y in zip(preds.cpu().numpy(), ys.cpu().numpy()):
    #         f.write(f"{pred}\t{y}\n")
def evaluate(args):
    config = read_yaml(args.config)
    dataset_name = config["dataset"]

    if dataset_name == "DFD":
        evaluate_DFD(args, config)
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not implemented")

if __name__ == '__main__':
    parser = argparse.ArgumentParser("DFD evaluation")
    parser.add_argument("--config", default='/home/wyj/FF++/config/DFD/DFD_marlin_CelebDF_lp.yaml',type=str, help="Path to DFD evaluation config file.")
    parser.add_argument("--data_path", default='/home/wyj/FF++/DFD',type=str, help="Path to DFD dataset.")
    parser.add_argument("--label_file", default='/home/wyj/FF++/DFD/label_DFD.txt',type=str, help="Path to label.txt.")
    parser.add_argument("--marlin_ckpt", type=str, default=None,
        help="Path to MARLIN checkpoint. Default: None, load from online.")
    parser.add_argument("--n_gpus", type=int, default=1)
    parser.add_argument("--precision", type=str, default="32")
    parser.add_argument("--num_workers", type=int, default=4)# 8->4
    parser.add_argument("--batch_size", type=int, default=16)# 32->16
    parser.add_argument("--epochs", type=int, default=2000, help="Max epochs to train.")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training.")

    args = parser.parse_args()

    evaluate(args)
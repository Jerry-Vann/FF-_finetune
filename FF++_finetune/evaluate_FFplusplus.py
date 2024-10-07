##用于进行FF++上finetune .git

import argparse

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torchmetrics import AUROC
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter

from dataset.FFplusplus import FFplusplusDataModule
from marlin_pytorch.config import resolve_config
from marlin_pytorch.util import read_yaml
from model.classifier import Classifier
from util.earlystop_lr import EarlyStoppingLR
from util.lr_logger import LrLogger
from util.seed import Seed
from util.system_stats_logger import SystemStatsLogger
from torchmetrics.classification import BinaryAUROC

def train_FFplusplus(args, config):
    writer = SummaryWriter(log_dir=f"logs/{config['model_name']}FT")  # TensorBoard 日志目录
    data_path = args.data_path
    resume_ckpt = args.resume
    n_gpus = args.n_gpus
    max_epochs = args.epochs
    label_file = args.label_file

    finetune = config["finetune"]
    learning_rate = config["learning_rate"]
    task = config["task"]

    if task == "appearance":
        # num_classes = 40
        num_classes = 1
    elif task == "action":
        num_classes = 35
    else:
        raise ValueError(f"Unknown task {task}")

    if finetune:
        backbone_config = resolve_config(config["backbone"])

        model = Classifier( #task从multilabel改为binary
            num_classes, config["backbone"], True, args.marlin_ckpt, "binary", config["learning_rate"],
            args.n_gpus > 1,
        )

        dm = FFplusplusDataModule(
            data_path, finetune, task, label_file,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            clip_frames=backbone_config.n_frames,
            temporal_sample_rate=2
        )

    else:
        model = Classifier( #task从multilabel改为binary
            num_classes, config["backbone"], False,
            None, "binary", config["learning_rate"], args.n_gpus > 1,
        )

        dm = FFplusplusDataModule(
            data_path, finetune, task, label_file,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            feature_dir=config["backbone"] + "_FF++",
            temporal_reduction=config["temporal_reduction"]
        )

    if args.skip_train:
        dm.setup()
        return resume_ckpt, dm

    strategy = None if n_gpus <= 1 else "ddp"
    accelerator = "cpu" if n_gpus == 0 else "gpu"

    ckpt_filename = config["model_name"] + "-{epoch}-{val_auc:.3f}"
    ckpt_monitor = "val_auc"

    try:
        precision = int(args.precision)
    except ValueError:
        precision = args.precision

    ckpt_callback = ModelCheckpoint(dirpath=f"ckpt/{config['model_name']}", save_last=True,
        filename=ckpt_filename,
        monitor=ckpt_monitor,
        mode="max")

    trainer = Trainer(log_every_n_steps=1, devices=n_gpus, accelerator=accelerator, benchmark=True,
        logger=True, precision=precision, max_epochs=max_epochs,
        strategy=strategy, resume_from_checkpoint=resume_ckpt,
        callbacks=[ckpt_callback, LrLogger(), EarlyStoppingLR(1e-6), SystemStatsLogger()])

    trainer.fit(model, dm)
    writer.close()  # 关闭 TensorBoard 日志
    return ckpt_callback.best_model_path, dm

def evaluate_FFplusplus(args, ckpt, dm):
    print("Load checkpoint", ckpt)
    model = Classifier.load_from_checkpoint(ckpt)
    # model = Classifier.load_from_checkpoint("checkpoint/MARLIN_FF++.ckpt") # 跳过训练
    accelerator = "cpu" if args.n_gpus == 0 else "gpu"
    trainer = Trainer(log_every_n_steps=1, devices=1 if args.n_gpus > 0 else 0, accelerator=accelerator, benchmark=True,
        logger=False, enable_checkpointing=False)
    Seed.set(42)
    model.eval()

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

    # 将预测结果保存到 txt 文件（修改3）
    # with open("predictions.txt", "w") as f:
    #     f.write("Predictions\tGround Truth\n")
    #     for pred, y in zip(preds.cpu().numpy(), ys.cpu().numpy()):
    #         f.write(f"{pred}\t{y}\n")

def evaluate(args):
    config = read_yaml(args.config)
    dataset_name = config["dataset"]

    if dataset_name == "FFplusplus":
        ckpt, dm = train_FFplusplus(args, config)
        evaluate_FFplusplus(args, ckpt, dm)
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not implemented")

if __name__ == '__main__':
    parser = argparse.ArgumentParser("FFplusplus evaluation")
    parser.add_argument("--config", default='/home/wyj/FF++_finetune/config/FFplusplus/marlin_FFplusplus_ft.yaml',type=str, help="Path to CelebDF evaluation config file.")
    parser.add_argument("--data_path", default='/home/wyj/FF++_finetune/FFplusplus',type=str, help="Path to FFplusplus dataset.")
    parser.add_argument("--label_file", default='/home/wyj/FF++_finetune/FFplusplus/label_FF++.txt',type=str, help="Path to label.txt.")
    parser.add_argument("--marlin_ckpt", type=str, default=None,
        help="Path to MARLIN checkpoint. Default: None, load from online.")
    parser.add_argument("--n_gpus", type=int, default=1)
    parser.add_argument("--precision", type=str, default="32")
    parser.add_argument("--num_workers", type=int, default=4)# 8->4
    parser.add_argument("--batch_size", type=int, default=8)# 32->16
    parser.add_argument("--epochs", type=int, default=2000, help="Max epochs to train.")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training.")
    parser.add_argument("--skip_train", action="store_true", default=False,
        help="Skip training and evaluate only.")

    args = parser.parse_args()
    if args.skip_train: # 跳过训练注释
        assert args.resume is not None

    evaluate(args)
from typing import Optional, Union, Sequence, Dict, Literal, Any

from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import CrossEntropyLoss, Linear, Identity, BCEWithLogitsLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import Accuracy, AUROC
from torch.utils.tensorboard import SummaryWriter
import torch

from marlin_pytorch import Marlin
from marlin_pytorch.config import resolve_config


class Classifier(LightningModule):
    # num_classes：分类的类别数量/backbone：骨干模型的名称/finetune：布尔值，指示是否对骨干模型进行微调/marlin_ckpt：骨干模型的检查点文件的可选路径
    def __init__(self, num_classes: int, backbone: str, finetune: bool,
        marlin_ckpt: Optional[str] = None,
        task: Literal["binary", "multiclass", "multilabel"] = "binary",
        learning_rate: float = 1e-4, distributed: bool = False
    ):
        super().__init__()# 调用父类的构造方法，初始化继承自LightningModule的属性。
        self.save_hyperparameters()#save_hyperparameters()方法保存构造函数中的所有参数作为模型的超参数

        if finetune:
            if marlin_ckpt is None:
                self.model = Marlin.from_online(backbone).encoder
            else:
                self.model = Marlin.from_file(backbone, marlin_ckpt).encoder
        else:
            self.model = None
        # 返回一个MarlinConfig类型，此类定义在config.py中，包含图像尺寸、补丁尺寸、编码器解码器相关各项参数等等
        config = resolve_config(backbone)
        # 定义一个全连接层，输入特征维度为config中的encoder_embed_dim(768)，输出类别数量为num_classes(40)
        self.fc = Linear(config.encoder_embed_dim, num_classes)

        self.learning_rate = learning_rate
        self.distributed = distributed
        self.task = task
        if task in "binary":
            self.loss_fn = BCEWithLogitsLoss()
            self.acc_fn = Accuracy(task=task, num_classes=1)
            self.auc_fn = AUROC(task=task, num_classes=1)
        elif task == "multiclass":
            self.loss_fn = CrossEntropyLoss()
            self.acc_fn = Accuracy(task=task, num_classes=num_classes)
            self.auc_fn = AUROC(task=task, num_classes=num_classes)
        elif task == "multilabel":
            self.loss_fn = BCEWithLogitsLoss()
            self.acc_fn = Accuracy(task="binary", num_classes=1)
            self.auc_fn = AUROC(task="binary", num_classes=1)

        self.writer = SummaryWriter(log_dir=f"logs/{backbone}")  # TensorBoard 日志目录

    @classmethod
    def from_module(cls, model, learning_rate: float = 1e-4, distributed=False):
        return cls(model, learning_rate, distributed)

    def forward(self, x):
        print(f"Input shape: {x.shape}, dtype: {x.dtype}")
        if self.model is not None:
            feat = self.model.extract_features(x, True)# 提取特征
        else:
            feat = x
        return self.fc(feat)# 通过全连接层进行分类

    def step(self, batch: Optional[Union[Tensor, Sequence[Tensor]]]) -> Dict[str, Tensor]:
        x, y = batch# 输入数据和标签
        y_hat = self(x)# 模型预测
        # loss = self.loss_fn(y_hat, y.float())# 计算损失
        if y.dim() == 1: # 如果 y 是一维张量（即只有一个维度），那么将其转换为二维张量
            y = y.unsqueeze(1)
        loss = self.loss_fn(y_hat, y.float())  # 计算损失

        prob = y_hat.sigmoid()# 预测结果经过sigmoid函数转换为概率
        acc = self.acc_fn(prob, y)# 计算准确率
        auc = self.auc_fn(prob, y)# 计算AUC
        return {"loss": loss, "acc": acc, "auc": auc}

    def training_step(self, batch: Optional[Union[Tensor, Sequence[Tensor]]] = None, batch_idx: Optional[int] = None,
        optimizer_idx: Optional[int] = None, hiddens: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        loss_dict = self.step(batch)# 单个训练步骤
        self.log_dict({f"train_{k}": v for k, v in loss_dict.items()}, on_step=True, on_epoch=True,
            prog_bar=False, sync_dist=self.distributed)# 记录训练指标
        self.writer.add_scalars("Metrics/Training", loss_dict, self.current_epoch)  # 记录训练指标到TensorBoard
        return loss_dict["loss"]# 返回损失值

    def validation_step(self, batch: Optional[Union[Tensor, Sequence[Tensor]]] = None, batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None
    ) -> Dict[str, Tensor]:
        loss_dict = self.step(batch)# 单个验证步骤
        self.log_dict({f"val_{k}": v for k, v in loss_dict.items()}, on_step=True, on_epoch=True,
            prog_bar=True, sync_dist=self.distributed)# 记录验证指标
        self.writer.add_scalars("Metrics/Validation", loss_dict, self.current_epoch)  # 记录验证指标到TensorBoard
        return loss_dict["loss"]# 返回损失值

    def predict_step(self, batch: Tensor, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
        return self(batch[0])# 预测步骤，返回模型预测结果

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate, betas=(0.5, 0.9))
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer, factor=0.5, patience=7, verbose=True, min_lr=1e-8),
                "monitor": "train_loss"# 监控训练损失
            }
        }# 返回优化器和学习率调度器的配置信息

    #Tensorboard添加内容
    def on_train_end(self):
        self.writer.close()  # 关闭 TensorBoard 日志
    # 在 PyTorch Lightning 中，on_train_end 是一个回调钩子，它自动在训练过程结束时被调用。
    # 这意味着你不需要在代码中显式调用这个函数；PyTorch Lightning 的内部机制会在训练结束时自动调用它。
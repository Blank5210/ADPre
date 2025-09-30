import torch.nn as nn
import torch
import numpy as np
import math
from scipy import special

from lightning_resgistry.models.builder import MODELS, build_model
from lightning_resgistry.models.losses.builder import build_criteria
import pytorch_lightning as pl
import torch.optim.lr_scheduler as lrs
from torch.nn import functional as F


@MODELS.register_module()
class MInterface(pl.LightningModule):
    def __init__(self, model, loss, optimizer, lr_scheduler, checkpoint_path=None):
        super().__init__()
        self.save_hyperparameters()
        self.model = build_model(self.hparams.model)
        self.checkpoint_path = checkpoint_path
        self.loss_function = None
        self.configure_loss()
        self.epoch_grad_norms = []

    def forward(self, img, label=None):
        if label is None:
            return self.model(img)
        else:
            return self.model(img, label)

    def training_step(self, batch, batch_idx):
        img, labels = batch
        # print(img.device, labels.device, self.device)
        # 如果损失函数在模型外定义
        if self.loss_function is not None:
            logits = self(img)
            loss = self.loss_function(logits, labels)
        # 损失函数在模型内定义
        else:
            logits, loss = self(img, labels)

        # 分类任务：预测类别
        preds = logits.argmax(dim=1)
        acc = (preds == labels).float().mean()

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        img, labels = batch
        if self.loss_function is not None:
            logits = self(img)
            loss = self.loss_function(logits, labels)
        else:
            logits, loss = self(img, labels)

        label_digit = labels
        logits_digit = logits.argmax(axis=1)

        correct_num = sum(label_digit == logits_digit).cpu().item()

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', correct_num/len(logits_digit),
                 on_step=False, on_epoch=True, prog_bar=True)

        return (correct_num, len(logits_digit))

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def on_validation_epoch_end(self):
        # Make the Progress Bar leave there
        self.print('')

    def configure_optimizers(self):
        if hasattr(self.hparams.optimizer, 'weight_decay'):
            weight_decay = self.hparams.optimizer.weight_decay
        else:
            weight_decay = 0
        if self.hparams.optimizer['type'] == 'Adam':
            optimizer = torch.optim.Adam(
                self.parameters(), lr=self.hparams.optimizer["lr"], weight_decay=weight_decay)
        else:
            optimizer = torch.optim.SGD(
                self.parameters(), lr=self.hparams.optimizer["lr"], weight_decay=weight_decay, momentum=0.0)

        if self.hparams.lr_scheduler is None:
            return optimizer
        else:
            if self.hparams.lr_scheduler["type"] == 'step':
                scheduler = lrs.StepLR(optimizer,
                                       step_size=self.hparams.lr_scheduler["step_size"],
                                       gamma=self.hparams.lr_scheduler["gamma"],
                                       )
            elif self.hparams.lr_scheduler["type"] == 'cosine':
                scheduler = lrs.CosineAnnealingLR(optimizer,
                                                  T_max=self.hparams.lr_scheduler["step_size"],
                                                  eta_min=self.hparams.lr_scheduler["min_lr"])
            else:
                raise ValueError('Invalid lr_scheduler type!')
            return [optimizer], [scheduler]

    def configure_loss(self):
        if hasattr(self.hparams, "loss"):
            if self.hparams.loss is not None:
                self.loss_function = nn.CrossEntropyLoss()

    def on_train_start(self):
        if self.checkpoint_path is not None:
            # 加载权重
            if self.checkpoint_path.endswith(".pth"):
                self.model.load_state_dict(torch.load(self.checkpoint_path))
            elif self.checkpoint_path.endswith(".ckpt"):
                checkpoint = torch.load(self.checkpoint_path)
                self.model.load_state_dict(checkpoint["state_dict"])

    def on_train_batch_end(self, batch, batch_idx, unused=None):
        total_norm = 0.0
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                total_norm += grad_norm ** 2
                # 记录每层梯度
                self.logger.experiment.add_scalar(f'grad_norm/{name}', grad_norm, self.global_step)

        total_norm = total_norm ** 0.5
        self.epoch_grad_norms.append(total_norm)
        # 记录总梯度
        self.logger.experiment.add_scalar('grad_norm/total', total_norm, self.global_step)

    # 在每个 epoch 结束后计算平均梯度并输出
    def on_train_epoch_end(self):
        if self.epoch_grad_norms:
            avg_grad = sum(self.epoch_grad_norms) / len(self.epoch_grad_norms)
            print(f"Epoch {self.current_epoch} avg grad norm: {avg_grad:.6f}")
            # 也可以写入 TensorBoard
            self.logger.experiment.add_scalar('grad_norm/epoch_avg', avg_grad, self.current_epoch)
            self.epoch_grad_norms.clear()
        # torch.cuda.empty_cache() 清除显存
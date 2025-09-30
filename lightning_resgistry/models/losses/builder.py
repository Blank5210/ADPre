"""
Criteria Builder

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.registry import Registry

LOSSES = Registry("losses")


class Criteria(object):
    def __init__(self, cfg=None):
        self.cfg = cfg if cfg is not None else []
        self.criteria = []
        self.build_losses()

    def build_losses(self):
        for loss_cfg in self.cfg:
            if loss_cfg['type'] == "CrossEntropyLoss":
                self.criteria.append(nn.CrossEntropyLoss())
            elif loss_cfg['type'] == "MSELoss":
                self.criteria.append(nn.MSELoss())
            else:
                raise NotImplementedError(f"Unknown loss type {loss_cfg.type}")

    def __call__(self, pre, labels):
        if len(self.criteria) == 0:
            # loss computation occurs in model
            return pre

        assert isinstance(pre, (list, tuple)), "pre must be a list or tuple"
        assert isinstance(labels, (list, tuple)), "labels must be a list or tuple"
        assert len(pre) == len(self.criteria), "Number of preds must match number of criteria"
        assert len(labels) == len(self.criteria), "Number of labels must match number of criteria"

        loss_list = []
        for index, c in enumerate(self.criteria):
            l = c(pre[index], labels[index])
            loss_list.append(l.view(1))  # 保证是 [1] 形状

        # loss = torch.mean(torch.cat(loss_list))  # 算术平均
        loss = torch.exp(torch.mean(torch.log(torch.cat(loss_list))))  # 几何平均

        return loss_list[0]



def build_criteria(cfg):
    return Criteria(cfg)

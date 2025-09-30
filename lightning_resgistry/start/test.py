import os
import torch
from mmengine.config import Config
from pytorch_lightning import Trainer
from lightning_resgistry.datasets.data_interface import build_data_interface
from lightning_resgistry.models import build_model


def main():
    # 加载配置文件
    config_file = "../configs/adimage/resnet.py"
    cfg = Config.fromfile(config_file)

    # 构建数据接口
    data_module = build_data_interface(cfg.data)

    # 构建模型
    model = build_model(cfg.model)

    # 加载模型权重
    checkpoint_path = "../checkpoints/model.ckpt"  # 替换为实际的权重文件路径
    model = model.load_from_checkpoint(checkpoint_path, **cfg.model)

    # 初始化 Trainer
    trainer = Trainer(**cfg.trainer)

    # 开始测试
    trainer.test(model, datamodule=data_module)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 设置可见的 GPU
    main()
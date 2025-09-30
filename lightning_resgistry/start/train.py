import os
import yaml

from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything
from lightning_resgistry.datasets.data_interface import build_data_interface
from lightning_resgistry.models.model_interface import build_model
import importlib.util


def load_py_config(path):
    spec = importlib.util.spec_from_file_location("config_module", path)
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    return cfg


def main():
    # 加载配置文件
    config_file = "../configs/adimage/unet-resnet.py"
    cfg = load_py_config(config_file)

    # set seed
    if cfg.seed is not None:
        seed_everything(cfg.seed, workers=True)

    # 构建数据接口
    data_module = build_data_interface(cfg.data)

    # 构建模型
    model = build_model(cfg.model)

    # 初始化 Trainer
    trainer = Trainer(**cfg.trainer)

    # 开始训练
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 设置可见的 GPU
    main()

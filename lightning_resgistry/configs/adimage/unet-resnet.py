from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger


save_dir = "D:/pythonProject/pytorch-lightning-template/lightning_resgistry/checkpoints"
seed = 42
logger = TensorBoardLogger(
        save_dir=save_dir,
        name="unet-resnet"
    )
checkpoint_callback = ModelCheckpoint(
                            dirpath=logger.log_dir,
                            monitor="val_loss",
                            mode="min",
                            save_top_k=3,
                            save_last=True,
                            filename="{epoch}-{val_loss:.2f}"
                        )

trainer = dict(
    # 基础训练控制
    max_epochs=100,
    min_epochs=1,
    max_steps=-1,
    limit_train_batches=1.0,   # 1.0 表示全部批次，可为 float 或 int
    limit_val_batches=1.0,
    limit_test_batches=1.0,
    val_check_interval=1.0,    # 1.0 表示每轮验证，可为 step 数

    # 硬件加速
    accelerator="gpu",          # {cpu, gpu, tpu, auto}
    devices=1,
    precision=32,               # {16, 32, 64, "16-mixed"}
    strategy="auto",             # {ddp, dp, ddp_spawn, etc, None}.

    # 日志
    logger=logger,
    log_every_n_steps=50,
    enable_progress_bar=True,

    # checkpoint & 回调
    enable_checkpointing=True,
    callbacks=[
        checkpoint_callback,
        EarlyStopping(
            monitor="train_loss",
            patience=50,
            mode="min"
        )
    ],

    # 梯度与优化
    gradient_clip_val=0.0,
    accumulate_grad_batches=1,
    benchmark=True,
    deterministic=False,
    reload_dataloaders_every_n_epochs=0,
)


# ===========================
# 数据集相关参数
test_size = 0.8
val_size = 0.1
# ===========================
data = dict(
    type="DInterface",

    train=dict(
        type="ADDataset",
        data_root="D:/DATA/train_data/train",
        split="train",
        random_seed=seed,

        augment=True,
        augmenter=dict(
            type="Augment3D",
            apply_rotate=False,

            apply_crop=True,
            crop_size=(140, 180, 140),

            apply_elastic=False,

            apply_normalize=True,

            apply_downsample=True,

            seed=seed
        ),
        test_size=test_size,
        val_size=val_size
    ),
    val=dict(
            type="ADDataset",
            data_root="D:/DATA/train_data/train",
            split="val",
            random_seed=seed,
            augment=True,
            augmenter=dict(
                type="Augment3D",
                apply_rotate=False,

                apply_crop=True,
                crop_size=(140, 180, 140),

                apply_elastic=False,

                apply_normalize=True,

                apply_downsample=True,

                seed=seed
            ),
            test_size=test_size,
            val_size=val_size
    ),
    test=dict(
        type="ADDataset",
        data_root="D:/DATA/train_data/train",
        split="test",
        random_seed=seed,
        augment=False,
        augmenter=dict(
            type="Augment3D",
            apply_rotate=False,

            apply_crop=True,
            crop_size=(140, 180, 140),

            apply_elastic=False,

            apply_normalize=True,

            apply_downsample=True,

            seed=seed
        ),
        test_size=test_size,
        val_size=val_size
    ),
    batch_size=4,
    num_workers=4
)


model = dict(
    type="MInterface",
    model=dict(
        type="unet-resnet",
        T=1000,
        ch=64,
        ch_mult=[1, 2, 4, 8],
        attn=[-1],
        num_res_blocks=1,
        dropout=0.1
    ),
    loss=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1)
    ],
    optimizer=dict(
        type="Adam",
        lr=1e-2,
        weight_decay=1e-6),
    lr_scheduler=dict(
        type="step",           # step, cosine
        step_size=40,
        gamma=0.5,
        min_lr=1e-5
    )
)
# ===========================
# 优化器相关参数
# ===========================


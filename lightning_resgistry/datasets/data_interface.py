import inspect
import importlib
import pickle as pkl
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

from .builder import DATASETS, DATA_INTERFACE, build_dataset, build_data_interface


@DATA_INTERFACE.register_module()
class DInterface(pl.LightningDataModule):

    def __init__(self,
                 train,
                 val,
                 test,
                 batch_size,
                 num_workers=8
                 ):
        super().__init__()
        self.save_hyperparameters()
        self.num_workers = num_workers
        self.train_cfg = train
        self.val_cfg = val
        self.test_cfg = test
        self.batch_size = batch_size

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.trainset = build_dataset(self.train_cfg)
            self.valset = build_dataset(self.val_cfg)

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.testset = build_dataset(self.test_cfg)

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.testset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

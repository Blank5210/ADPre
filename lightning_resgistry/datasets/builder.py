from mmengine.registry import Registry

DATASETS = Registry("datasets")
DATA_INTERFACE = Registry("data_interface")
AUGMENT = Registry("data_augment")


def build_dataset(cfg):
    """Build datasets."""
    return DATASETS.build(cfg)


def build_data_interface(cfg):
    return DATA_INTERFACE.build(cfg)


def build_data_augmenter(cfg):
    return AUGMENT.build(cfg)

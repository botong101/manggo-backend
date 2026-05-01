import dataclasses

MIN_IMAGES_PER_CLASS = 5


@dataclasses.dataclass
class RetrainConfig:
    epochs:                  int   = 10
    learning_rate:           float = 1e-4
    batch_size:              int   = 16
    val_split:               float = 0.2
    unfreeze_top_n_layers:   int   = 20
    early_stopping_patience: int   = 3
    lr_reduce_factor:        float = 0.5
    lr_reduce_patience:      int   = 2
    min_images_per_class:    int   = MIN_IMAGES_PER_CLASS

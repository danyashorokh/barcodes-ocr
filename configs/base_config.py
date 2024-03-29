import json
import os
import typing as tp
from dataclasses import asdict, dataclass, field
from pytorch_lightning.callbacks import Callback

import albumentations as albu
import torch
from torch.optim.optimizer import Optimizer


@dataclass
class Config:
    num_workers: int
    seed: int
    loss: torch.nn.Module
    optimizer: type(Optimizer)
    optimizer_kwargs: tp.Mapping
    warmup_iter: int
    scheduler: tp.Optional[tp.Any]
    scheduler_kwargs: tp.Mapping
    preprocessing: tp.Callable
    img_size: tp.Tuple[int, int]
    vocab: str
    max_length: int
    expand_char: str
    augmentations: albu.Compose
    batch_size: int
    n_epochs: int
    backbone_name: str
    cnn_backbone_pretrained: bool
    cnn_output_size: int
    rnn_features_num: int
    rnn_dropout: float
    rnn_bidirectional: bool
    rnn_num_layers: int
    num_classes: int
    experiment_name: str
    log_metrics: tp.List[str]
    valid_metric: str
    minimize_metric: bool
    images_dir: str
    train_dataset_path: str
    valid_dataset_path: str
    test_dataset_path: str
    project_name: str
    checkpoints_dir: str = field(init=False)
    callbacks: tp.List[Callback] = field(default_factory=list)
    num_iteration_on_epoch: int = 0
    checkpoint_name: tp.Optional[str] = None
    trainer_kwargs: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        res = {}
        for k, v in asdict(self).items():
            try:
                if isinstance(v, torch.nn.Module):
                    res[k] = v.__class__.__name__
                elif isinstance(v, dict):
                    res[k] = json.dumps(v, indent=4)
                else:
                    res[k] = str(v)
            except Exception:
                res[k] = str(v)
        return res

    def __post_init__(self):
        self.checkpoints_dir = os.path.join(  # noqa: WPS601
            './weights',
            self.experiment_name,
        )

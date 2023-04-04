
import argparse
import logging
from typing import Any
from runpy import run_path

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, RichProgressBar
from clearml import Task

from configs.base_config import Config
from src.model import CRNN
from src.utils import set_global_seed
from src.metrics import accuracy
from src.dataset import get_loaders
from src.data_module import DataModule
from src.train_module import TrainModule


def arg_parse() -> Any:
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', type=str, help='config file')
    return parser.parse_args()


def train(config: Config):

    pl.seed_everything(config.seed)

    # init logger
    task = Task.init(project_name=config.project_name, task_name=config.experiment_name)
    task.connect(config.to_dict())

    loaders = get_loaders(config)

    # init model
    model = CRNN(
        cnn_backbone_name=config.backbone_name,
        cnn_backbone_pretrained=config.cnn_backbone_pretrained,
        cnn_output_size=config.cnn_output_size,
        rnn_features_num=config.rnn_features_num,
        rnn_dropout=config.rnn_dropout,
        rnn_bidirectional=config.rnn_bidirectional,
        rnn_num_layers=config.rnn_num_layers,
        num_classes=config.num_classes,
    )
    if config.checkpoint_name is not None:
        model.load_state_dict(torch.load(config.checkpoint_name)['my_state_dict'])

    model = model.train()

    optimizer = config.optimizer(params=model.parameters(), **config.optimizer_kwargs)
    if config.scheduler is not None:
        scheduler = config.scheduler(optimizer=optimizer, **config.scheduler_kwargs)
    else:
        scheduler = None

    # init callbacks
    model_checkpoint = ModelCheckpoint(
        dirpath=config.checkpoints_dir, filename='{epoch}_{val_loss:.2f}',
        monitor=config.valid_metric, verbose=False, save_last=None,
        save_top_k=3, save_weights_only=True, mode='min' if config.minimize_metric else 'max')
    model_checkpoint.FILE_EXTENSION = '.pth'

    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)

    callbacks = [
        model_checkpoint,
        early_stopping,
        lr_monitor,
        RichProgressBar(leave=False),
    ]

    data = DataModule(loaders)
    model = TrainModule(model, config.loss, accuracy, optimizer, scheduler)

    trainer = pl.Trainer(
        max_epochs=config.n_epochs,
        callbacks=callbacks,
        **config.trainer_kwargs,
    )
    trainer.fit(model, data)

    trainer.test(ckpt_path=model_checkpoint.best_model_path, datamodule=data)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    args = arg_parse()
    config_module = run_path(args.config_file)
    config = config_module['config']

    set_global_seed(config.seed)
    train(config)

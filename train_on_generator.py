
import argparse
import logging
from typing import Any
from runpy import run_path


import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, RichProgressBar

from clearml import Task

from src.model import CRNN
from src.generator.dataset_generator import BarcodeDataset
from src.base_config import Config
from src.utils import set_global_seed

from torchmetrics import CharErrorRate
from torchmetrics import MetricCollection


def arg_parse() -> Any:
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', type=str, help='config file')
    return parser.parse_args()


def train(config: Config):

    pl.seed_everything(config.seed)

    # init logger
    task = Task.init(project_name=config.project_name, task_name=config.experiment_name)
    task.connect(config.to_dict())
    # clearml_logger = task.get_logger()

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
    model = model.train()

    barcode_loader = torch.utils.data.DataLoader(
        BarcodeDataset(epoch_size=config.num_iteration_on_epoch, vocab='0123456789 '), batch_size=config.batch_size,
    )

    optimizer = config.optimizer(params=model.parameters(), **config.optimizer_kwargs)
    if config.scheduler is not None:
        scheduler = config.scheduler(optimizer=optimizer, **config.scheduler_kwargs)
    else:
        scheduler = None

    # init metrics
    metrics = MetricCollection({
        'сer': CharErrorRate(),
    })

    # init callbacks
    model_checkpoint = ModelCheckpoint(
        dirpath=config.checkpoints_dir, filename='{epoch}_{val_loss:.2f}_{val_f1:.2f}',
        monitor=config.valid_metric, verbose=False, save_last=None,
        save_top_k=10, save_weights_only=True, mode='min' if config.minimize_metric else 'max')
    model_checkpoint.FILE_EXTENSION = '.pth'

    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)

    callbacks = [
        model_checkpoint,
        early_stopping,
        lr_monitor,
        RichProgressBar(leave=False),
    ]

    class DataModule(pl.LightningDataModule):
        def __init__(self, loader):
            super(DataModule, self).__init__()
            self.loader = loader

        def train_dataloader(self):
            return self.loader

        def val_dataloader(self):
            return self.loader

        # def test_dataloader(self):
        #     return self.loader

    class TrainModule(pl.LightningModule):
        def __init__(self, model, loss, metrics, optimizer, scheduler):
            super(TrainModule, self).__init__()

            self.model = model
            self.loss = loss
            self.optimizer = optimizer
            self.scheduler = scheduler

            # metrics
            self.train_metrics = metrics.clone(prefix='train_')
            self.val_metrics = metrics.clone(prefix='val_')
            self.save_hyperparameters()

        def forward(self, x: torch.Tensor):
            """Get output from model.

            Args:
                x: torch.Tensor - batch of images.

            Returns:
                output: torch.Tensor - predicted logits.
            """
            return self.model(x)

        def training_step(self, batch, batch_idx):

            input_images, targets, target_lengths = batch
            output = self(input_images)

            input_lengths = [output.size(0) for _ in input_images]
            input_lengths = torch.LongTensor(input_lengths)
            target_lengths = torch.flatten(target_lengths)

            loss = self.loss(output, targets, input_lengths, target_lengths)
            # self.train_metrics.update(output, targets)

            self.log('train_loss', loss, prog_bar=True, logger=True, on_step=True)

            return loss

        # def training_epoch_end(self, outputs):

        #     train_metrics = self.train_metrics.compute()

        def validation_step(self, batch, batch_idx):

            input_images, targets, target_lengths = batch
            output = self(input_images)

            input_lengths = [output.size(0) for _ in input_images]
            input_lengths = torch.LongTensor(input_lengths)
            target_lengths = torch.flatten(target_lengths)

            loss = self.loss(output, targets, input_lengths, target_lengths)
            # self.val_metrics.update(output, targets)

            self.log('val_loss', loss, prog_bar=True, logger=True, on_step=True)

            return loss

        def configure_optimizers(self):
            """To callback for configuring optimizers.

            Returns:
                optimizer: torch.optim - optimizer for PL.
            """
            return {'optimizer': self.optimizer, 'lr_scheduler': {'scheduler': self.scheduler}}

        def on_save_checkpoint(self, checkpoint):
            """Save custom state dict.

            Function is needed, because we want only timm state_dict for scripting.

            Args:
                checkpoint: pl.checkpoint - checkpoint from PL.
            """
            checkpoint['my_state_dict'] = self.model.state_dict()

    data = DataModule(barcode_loader)
    model = TrainModule(model, config.loss, metrics, optimizer, scheduler)

    trainer = pl.Trainer(
        max_epochs=config.n_epochs,
        callbacks=callbacks,
        **config.trainer_kwargs,
    )
    trainer.fit(model, data)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    args = arg_parse()
    config_module = run_path(args.config_file)
    config = config_module['config']

    set_global_seed(config.seed)
    train(config)

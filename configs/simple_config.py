from datetime import datetime
from functools import partial
import os

import albumentations as albu
import torch
from torch.nn import CTCLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR  # noqa

from configs.base_config import Config
from src.utils import preprocess_imagenet

SEED = 42
VOCAB = ' 0123456789'
MAX_LENGTH = 15
EXPAND_CHAR = ' '
IMG_SIZE = (512, 280)
BATCH_SIZE = 16
N_EPOCHS = 50
NUM_ITERATION_ON_EPOCH = 100
ROOT_PATH = os.path.join(os.environ.get('ROOT_PATH'))

augmentations = albu.Compose(
    [
        albu.HueSaturationValue(
            hue_shift_limit=20,
            sat_shift_limit=30,
            val_shift_limit=20,
            p=0.5,
        ),
        albu.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        albu.GaussianBlur(),
    ])


config = Config(
    num_workers=0,
    seed=SEED,
    loss=CTCLoss(zero_infinity=True),
    optimizer=torch.optim.Adam,
    optimizer_kwargs={
        'lr': 1e-3,
        'weight_decay': 5e-4,
    },
    warmup_iter=0,
    # scheduler=ReduceLROnPlateau, scheduler_kwargs={'mode': 'min', 'factor': 0.1, 'patience': 5},
    scheduler=MultiStepLR, scheduler_kwargs={'milestones': [5, 25], 'gamma': 0.1},
    img_size=IMG_SIZE,
    vocab=VOCAB,
    max_length=MAX_LENGTH,
    expand_char=EXPAND_CHAR,
    augmentations=augmentations,
    preprocessing=partial(preprocess_imagenet, img_size=IMG_SIZE),
    batch_size=BATCH_SIZE,
    num_iteration_on_epoch=NUM_ITERATION_ON_EPOCH,
    n_epochs=N_EPOCHS,
    backbone_name='resnet18d',
    cnn_backbone_pretrained=True,
    cnn_output_size=4608,
    rnn_features_num=128,
    rnn_dropout=0.1,
    rnn_bidirectional=True,
    rnn_num_layers=2,
    num_classes=len(VOCAB) + 1,
    log_metrics=[],
    valid_metric='val_loss',
    minimize_metric=True,
    images_dir=os.path.join(ROOT_PATH, 'full_dataset'),
    train_dataset_path=os.path.join(ROOT_PATH, 'train_df.csv'),
    valid_dataset_path=os.path.join(ROOT_PATH, 'valid_df.csv'),
    test_dataset_path=os.path.join(ROOT_PATH, 'test_df.csv'),
    project_name='[OCR]barcodes',
    experiment_name=f'{os.path.basename(__file__).split(".")[0]}_{datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}',
    trainer_kwargs={
        'accelerator': 'mps',
        'devices': 1,
    },
    checkpoint_name=None,
)

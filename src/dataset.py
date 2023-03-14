
import logging
import os
import typing as tp
from collections import OrderedDict

import albumentations as albu
import cv2
import numpy as np
import pandas as pd
from configs.base_config import Config
import torch
from torch.utils.data import DataLoader, Dataset


class BarcodeDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        config: Config,
        augmentation: tp.Optional[albu.Compose] = None,
        preprocessing: tp.Optional[tp.Callable] = None,
    ):
        self.df = df
        self.config = config
        self.vocab = self.config.vocab
        self.max_length = self.config.max_length
        self.expand_char = self.config.expand_char
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.char2index = dict((char, self.vocab.index(char) + 1) for char in self.vocab)

    def __getitem__(self, idx: int) -> tp.Dict[str, np.ndarray]:
        row = self.df.iloc[idx]

        value = row['result']
        orig_value_len = len(value)
        # expand target
        if len(value) < self.max_length:
            value += self.expand_char * (self.max_length - len(value))
        img_path = f'{os.path.join(self.config.images_dir, row.filename)}'

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.augmentation:
            image = self.augmentation(image=image)['image']
        if self.preprocessing:
            image = self.preprocessing(image)

        image = torch.FloatTensor(image).permute(2, 1, 0)

        return image, torch.LongTensor(self.encode_value(value)), torch.LongTensor([orig_value_len])

    def __len__(self) -> int:
        return len(self.df)

    def encode_value(self, value):
        return [self.char2index[char] for char in value]

    def decode_predict(self, predict):
        return ''.join(self.vocab[index - 1] for index in predict)


def _get_dataframes(
    config: Config,
) -> tp.Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(config.train_dataset_path)
    valid_df = pd.read_csv(config.valid_dataset_path)
    test_df = pd.read_csv(config.test_dataset_path)

    logging.info(f'Train dataset: {len(train_df)}')
    logging.info(f'Valid dataset: {len(valid_df)}')
    logging.info(f'Test dataset: {len(test_df)}')

    return train_df, valid_df, test_df


def get_datasets(config: Config) -> tp.Tuple[Dataset, Dataset, Dataset]:
    df_train, df_val, df_test = _get_dataframes(config)

    train_dataset = BarcodeDataset(
        df_train,
        config,
        augmentation=config.augmentations,
        preprocessing=config.preprocessing,
    )

    valid_dataset = BarcodeDataset(
        df_val,
        config,
        preprocessing=config.preprocessing,
    )

    test_dataset = BarcodeDataset(
        df_test,
        config,
        preprocessing=config.preprocessing,
    )

    return train_dataset, valid_dataset, test_dataset


def get_loaders(
    config: Config,
) -> tp.Tuple[tp.OrderedDict[str, DataLoader], tp.Dict[str, DataLoader]]:
    train_dataset, valid_dataset, test_dataset = get_datasets(config)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    return OrderedDict({'train': train_loader, 'valid': valid_loader, 'infer': test_loader})

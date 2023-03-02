import barcode  # pip install git+https://github.com/WhyNotHugo/python-barcode
import numpy as np
import torch

from barcode.writer import ImageWriter


class BarcodeDataset(torch.utils.data.Dataset):
    def __init__(self, epoch_size, vocab):
        self.epoch_size = epoch_size
        self.char2index = dict((char, vocab.index(char) + 1) for char in vocab)

    def __getitem__(self, i: int):
        value = str(np.random.randint(10e11, 10e12))
        # value = '12345678901'  # учим на одной картинке, чтобы убедиться, что сеть переобучается

        ean = barcode.UPCA(value, writer=ImageWriter())
        image = np.asarray(ean.render()) / 255.0
        image = torch.FloatTensor(image).permute(2, 0, 1)
        return image, torch.LongTensor(self.encode_value(value)), torch.LongTensor([len(value)])

    def __len__(self):
        return self.epoch_size

    def encode_value(self, value):
        return [self.char2index[char] for char in value]

    def decode_predict(self, predict):
        return ''.join(self.vocab[index - 1] for index in predict)

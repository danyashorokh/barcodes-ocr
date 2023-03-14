import barcode  # pip install git+https://github.com/WhyNotHugo/python-barcode
import numpy as np
import cv2
import torch

from barcode.writer import ImageWriter


class BarcodeDataset(torch.utils.data.Dataset):
    def __init__(self, epoch_size, vocab, max_length, img_size):
        self.epoch_size = epoch_size
        self.char2index = dict((char, vocab.index(char) + 1) for char in vocab)
        self.max_length = max_length
        self.img_size = img_size

    def __getitem__(self, i: int):
        value = str(np.random.randint(10 ** (self.max_length - 1), 10 ** self.max_length))
        # value = '012345678901'  # учим на одной картинке, чтобы убедиться, что сеть переобучается

        ean = barcode.EAN13(value, writer=ImageWriter())
        image = np.asarray(ean.render()) / 255.0
        # image = cv2.resize(image, self.img_size)
        image = torch.FloatTensor(image).permute(2, 0, 1)
        return image, torch.LongTensor(self.encode_value(value)), torch.LongTensor([len(value)])

    def __len__(self):
        return self.epoch_size

    def encode_value(self, value):
        return [self.char2index[char] for char in value]

    def decode_predict(self, predict):
        return ''.join(self.vocab[index - 1] for index in predict)

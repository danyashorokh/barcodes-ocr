import os
import random
from typing import Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps

MAX_PIXEL_INTENSITY = 255


def preprocess_imagenet(im: np.ndarray, img_size: Union[int, Tuple[int, int]]) -> np.ndarray:
    im = im.astype(np.float32)
    im /= MAX_PIXEL_INTENSITY
    if isinstance(img_size, int):
        target_size = (img_size, img_size)
    elif isinstance(img_size, tuple):
        target_size = img_size
    else:
        raise ValueError(f'bad img_size format {img_size}, one need int or tuple')
    im = cv2.resize(im, target_size)
    im = np.transpose(im, (2, 0, 1))
    im -= np.array([0.485, 0.456, 0.406])[:, None, None]
    im /= np.array([0.229, 0.224, 0.225])[:, None, None]
    return im


def set_global_seed(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def padding(img, expected_size):
    desired_size = expected_size
    delta_width = desired_size - img.size[0]
    delta_height = desired_size - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    return ImageOps.expand(img, padding)


def resize_with_padding(img, expected_size):
    img.thumbnail((expected_size[0], expected_size[1]))
    delta_width = expected_size[0] - img.size[0]
    delta_height = expected_size[1] - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    return ImageOps.expand(img, padding)


def preprocess_image(img: np.ndarray, img_size) -> np.ndarray:
    img = Image.fromarray(img)
    img = resize_with_padding(img, img_size)
    img = np.array(img)
    return img

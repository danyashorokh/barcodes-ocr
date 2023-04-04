
import typing as tp

import numpy as np
import torch
import torch.nn as nn

from src.utils import preprocess_imagenet, get_code


class BarcodeRecognizer:

    def __init__(self, model_path: str, device: str):
        self._model_path = model_path
        self._device = device

        self._model: nn.Module = torch.jit.load(self._model_path, map_location=self._device)
        self._size: tp.Tuple[int, int] = self._model.size
        self._vocab: str = self._model.vocab
        self._index2char = dict((self._vocab.index(char), char) for char in self._vocab)

    @property
    def vocab(self) -> tp.Dict:
        return self._index2char

    @property
    def size(self) -> tp.Tuple:
        return self._size

    def predict(self, image: np.ndarray) -> str:
        """Предсказание штрих-кода.

        :param image: RGB изображение;
        :return: штрих-код.
        """
        return self._postprocess_predict(self._predict(image))

    def _predict(self, image: np.ndarray) -> torch.Tensor:
        """Сырое предсказание.

        :param image: RGB изображение;
        :return: сырое предсказание.
        """
        batch = preprocess_imagenet(image, self._size)
        batch = torch.from_numpy(batch)[None]

        with torch.no_grad():
            pred = self._model(batch.to(self._device)).detach().cpu()

        return pred

    def _postprocess_predict(self, pred: torch.Tensor) -> str:
        """Постобработка для получения штрих-кода.

        :param pred: сырое предсказание модели;
        :return: штрих-код.
        """
        prediction = get_code(pred).numpy()
        if not len(prediction):
            return ''
        return ''.join([self._index2char[i] for i in prediction])

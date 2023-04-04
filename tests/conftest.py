
import os
import pytest

import cv2
import numpy as np

from tests.config_tests import Config
from src.wrapper import BarcodeRecognizer

TESTS_DIR = os.path.dirname(__file__)


@pytest.fixture(scope='session')
def dummy_input():
    return np.random.rand(*Config.image_size, 3)


@pytest.fixture(scope='session')
def sample_image_np():
    img = cv2.imread(os.path.join(TESTS_DIR, 'fixtures', 'images', '4604248003517.jpg'))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


@pytest.fixture(scope='session')
def model_wrapper():
    return BarcodeRecognizer(model_path=Config.weights, device=Config.device)

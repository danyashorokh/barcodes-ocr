
import os

TESTS_DIR = os.path.dirname(__file__)


class Config:
    weights = os.path.join(TESTS_DIR, '..', 'weights', 'model_ocr.zip')
    device = 'cpu'
    image_size = (512, 280)

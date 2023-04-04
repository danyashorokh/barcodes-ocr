
from copy import deepcopy

import numpy as np


def test_model_empty_tensor(model_wrapper, dummy_input):
    result = model_wrapper.predict(dummy_input)
    assert len(result) < 5


def test_model_np_image(model_wrapper, sample_image_np):
    result = model_wrapper.predict(sample_image_np)
    assert result == '4604248003517'


def test_predict_dont_mutate_orig_image(model_wrapper, sample_image_np):
    initial_image = deepcopy(sample_image_np)
    model_wrapper.predict(sample_image_np)
    assert np.allclose(initial_image, sample_image_np)

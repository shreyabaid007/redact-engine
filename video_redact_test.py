# test_performance.py
import time
import pytest
import numpy as np
from main import get_image_tensor, visualize


def test_image_tensor_conversion_performance():
    image = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)

    start_time = time.time()
    for _ in range(100):
        tensor = get_image_tensor(image)
    end_time = time.time()

    average_time = (end_time - start_time) / 100
    assert average_time < 0.1  # Should take less than 0.1 seconds per conversion


@pytest.mark.parametrize("image_size", [(480, 640), (720, 1280), (1080, 1920)])
def test_visualize_performance_scaling(image_size):
    image = np.random.randint(0, 255, (*image_size, 3), dtype=np.uint8)
    boxes = [[0, 0, 100, 100]] * 10

    start_time = time.time()
    result = visualize(image, boxes, 1.0)
    processing_time = time.time() - start_time

    # Processing time should scale roughly linearly with image size
    max_allowed_time = (image_size[0] * image_size[1]) / (480 * 640) * 0.1
    assert processing_time < max_allowed_time
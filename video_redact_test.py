# test_properties.py
from hypothesis import given, strategies as st
import numpy as np
from main import scale_box, visualize

@given(st.lists(st.floats(min_value=0, max_value=100), min_size=4, max_size=4),
       st.integers(min_value=50, max_value=1000),
       st.integers(min_value=50, max_value=1000),
       st.floats(min_value=0.1, max_value=5.0))
def test_scale_box_properties(box, width, height, scale):
    scaled = scale_box(box, width, height, scale)
    assert len(scaled) == 4
    assert all(0 <= x <= width for x in [scaled[0], scaled[2]])
    assert all(0 <= y <= height for y in [scaled[1], scaled[3]])

@given(st.integers(min_value=10, max_value=100),
       st.integers(min_value=10, max_value=100))
def test_visualize_properties(width, height):
    image = np.zeros((height, width, 3), dtype=np.uint8)
    boxes = [[0, 0, width//2, height//2]]
    result = visualize(image, boxes, 1.0)
    assert result.shape == image.shape
    assert result.dtype == image.dtype
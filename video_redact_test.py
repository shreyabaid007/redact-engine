# test_basic_utils.py
import unittest
import os
import numpy as np
from main import create_directory_if_not_exists, scale_box, get_device


class TestBasicUtils(unittest.TestCase):
    def test_directory_creation(self):
        test_path = "test_dir/test_file.txt"
        create_directory_if_not_exists(test_path)
        self.assertTrue(os.path.exists("test_dir"))
        os.rmdir("test_dir")  # Cleanup

    def test_scale_box(self):
        box = [10, 10, 20, 20]
        image_width = 100
        image_height = 100
        scale = 2.0
        scaled = scale_box(box, image_width, image_height, scale)
        self.assertEqual(len(scaled), 4)
        self.assertTrue(all(isinstance(x, float) for x in scaled))

    def test_get_device(self):
        device = get_device()
        self.assertIn(device, ['cuda', 'cpu'])


if __name__ == '__main__':
    unittest.main()
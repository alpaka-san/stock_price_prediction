import unittest
from models import Model
import numpy as np


class TestModel(unittest.TestCase):

    def test_io_shape(self):
        model = Model()
        data = np.arange(100).reshape(-1,5,1)
        self.assertTrue(model(data).shape == (20,5))

if __name__ == "__main__":
    unittest.main()
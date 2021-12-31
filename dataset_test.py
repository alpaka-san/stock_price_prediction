import unittest
from dataset import Dataset


class TestDataset(unittest.TestCase):

    def test_instantiation(self):
        stock_code = "3092.T"
        dataset = Dataset(stock_code)
        self.assertIsInstance(dataset, Dataset)


if __name__ == "__main__":
    unittest.main()
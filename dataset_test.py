import unittest
from dataset import Dataset, get_historical_data
import pandas as pd


class TestDataset(unittest.TestCase):

    def test_instantiation(self):
        stock_code = "3092.T"
        date = "2019-01-01"
        dataset = Dataset(stock_code, date)

        self.assertIsInstance(dataset, Dataset)

        self.assertEqual(len(dataset.train_data), 3)
        self.assertEqual(len(dataset.val_data), 3)
        self.assertEqual(len(dataset.test_data), 3)
        self.assertEqual(len(dataset.test_open_data), 2)

    def test_instantiation_without_code(self):
        with self.assertRaises(ValueError):
            dataset = Dataset()

    def test_get_historical_data_without_code(self):
        with self.assertRaises(ValueError):
            get_historical_data()

    def test_get_historical_data(self):
        data = get_historical_data("3092.T")
        self.assertIsInstance(data, pd.DataFrame)

if __name__ == "__main__":
    unittest.main()
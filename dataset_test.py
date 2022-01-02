import unittest
from dataset import Dataset, get_historical_data
import pandas as pd
import datetime
from operator import add
from functools import reduce


class TestDataset(unittest.TestCase):

    datalist = [
            [datetime.date(2018+y, 12 if y==0 else 1, k), k+y*31, (k+y*31)*10]
                for y in [0,1] for k in range(1,32)
        ]
    dummyData = pd.DataFrame(
        datalist,
        columns=["Date", "open", "Adj. Close"],
    )

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
    
    def test__split_data(self):
        dataset = Dataset("3092.T")
        ret = dataset._split_data(self.dummyData, "2019-01-01")

        train_X = ret[0]
        train_Y = ret[1]
        val_X = ret[2]
        val_Y = ret[3]
        test_X = ret[4]
        test_Y = ret[5]
        test_open_Y = ret[6]
        timestamp_train_Y = ret[7]
        timestamp_val_Y = ret[8]
        timestamp_test_Y = ret[9]

        self.assertTrue((train_X[5:] == train_Y[:-5]).all())
        self.assertTrue((val_X[5:] == val_Y[:-5]).all())
        self.assertTrue((test_X[5:] == test_Y[:-5]).all())

        self.assertTrue(test_open_Y.shape == test_Y.shape)
        self.assertTrue(timestamp_train_Y.shape[0] - 5 == reduce(add, train_Y.shape[:2]))
        #TODO: small overlap between val and test
        self.assertTrue(timestamp_val_Y.shape[0] == reduce(add, val_Y.shape[:2]))
        self.assertTrue(timestamp_test_Y.shape[0] == reduce(add, test_Y.shape[:2]))

if __name__ == "__main__":
    unittest.main()
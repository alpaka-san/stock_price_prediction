from yahoo_finance_api2 import share
from yahoo_finance_api2.exceptions import YahooFinanceError
import pandas as pd
import numpy as np


def get_historical_data(Code):
    if Code is None:
        raise ValueError("""Specify a stock code. """)

    share_data = share.Share(Code).get_historical(
        share.PERIOD_TYPE_YEAR,
        6,
        share.FREQUENCY_TYPE_DAY,
        1
    )
    columns = list(share_data.keys())
    columns[0] = "Date"
    columns[-2] = "Adj. Close"      # TODO: it's not actually Adjusted Close, but Close. 
    df = pd.DataFrame(
        list(zip(*share_data.values())), 
        columns=columns
    )
    df["Date"] = pd.to_datetime(df["Date"], unit="ms").dt.date
    return df


class Dataset():
    def __init__(
        self,
        Code: str = None,
        Date: str = "2018-01-01", 
        End_Date: str = "2021-12-31", # for debug
    ) -> None:
        df = get_historical_data(Code)

        #TODO: ad-hoc.
        train_X, train_Y, val_X, val_Y, test_X, test_Y, test_open_Y, timestamp_train_Y, timestamp_val_Y, timestamp_test_Y = self._split_data(df, Date)

        self.train_data = (train_X, train_Y, timestamp_train_Y)
        self.val_data = (val_X, val_Y, timestamp_val_Y)
        self.test_data = (test_X, test_Y, timestamp_test_Y)
        self.test_open_data = (test_open_Y, timestamp_test_Y)

    @staticmethod
    def _split_data(df, Date, input_length=5):
        # Date = "2016-01-01"
        df["Date"] = pd.to_datetime(df["Date"]) #TODO: this is just a WA
        data_train_raw = df["Adj. Close"][df["Date"] < Date].to_numpy()
        data_valtest_raw = df["Adj. Close"][df["Date"] >= Date].to_numpy()
        data_valtest_open = df["open"][df["Date"] >= Date].to_numpy()

        timestamp_train_Y = df["Date"][df["Date"] < Date]
        timestamp_valtest_Y = df["Date"][df["Date"] >= Date]
        timestamp_val_Y = timestamp_valtest_Y[:timestamp_valtest_Y.shape[0] // 2]
        timestamp_test_Y = timestamp_valtest_Y[timestamp_valtest_Y.shape[0] // 2:]

        def _data_reshape(data):
            reshaped = []
            for i in range(0, len(data) - (input_length+5)):
                reshaped.append(data[i : i + (input_length+5)].copy())

            reshaped_x = np.array([r[:input_length] for r in reshaped]).reshape((-1, input_length, 1))
            reshaped_y = np.array([r[input_length:] for r in reshaped]).reshape((-1, 5, 1))
            return reshaped_x, reshaped_y

        train_X, train_Y = _data_reshape(data_train_raw)
        valtest_X, valtest_Y = _data_reshape(data_valtest_raw)
        _, valtest_open_Y = _data_reshape(data_valtest_open)

        val_X = valtest_X[:valtest_X.shape[0] // 2]
        val_Y = valtest_Y[:valtest_Y.shape[0] // 2]
        test_X = valtest_X[valtest_X.shape[0] // 2:]
        test_Y = valtest_Y[valtest_Y.shape[0] // 2:]
        test_open_Y = valtest_open_Y[valtest_open_Y.shape[0] // 2:]

        return train_X, train_Y, val_X, val_Y, test_X, test_Y, test_open_Y, timestamp_train_Y, timestamp_val_Y, timestamp_test_Y


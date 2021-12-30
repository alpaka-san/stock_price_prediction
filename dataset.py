from yahoo_finance_api2 import share
from yahoo_finance_api2.exceptions import YahooFinanceError


class Dataset():
		def __init__(
				Code: str = None,
				Date: str = "2018-01-01", 
				End_Date: str = "2021-12-31", # for debug
		) -> None:
				single_data = share.Share("3092.T").get_historical( # zozo
				    share.PERIOD_TYPE_YEAR,
				    6,
				    share.FREQUENCY_TYPE_DAY,
				    1
				)
				# data preprocessing
				# split to train / val / test
				self.train_data = (train_X, train_Y, timestamp_train)
				self.val_data = (val_X, val_Y, timestamp_val)
				self.test_data = (test_X, test_Y, timestamp_test)
				# ***_X,Y consists of 
				# ([... Close value ...], [... Open value ...])
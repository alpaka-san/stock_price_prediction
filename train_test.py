import unittest
from train import main
import numpy as np
import hydra
import yaml
import os


class TestTrain(unittest.TestCase):

    if not os.path.exists("tmp"): #TODO: use dummyfile
        os.mkdir("tmp")

    def test_main(self):
        with hydra.initialize(config_path="."):
            cfg = hydra.compose(config_name="config.yaml")
            cfg.stock_code = "3092.T"
            cfg.training.epochs = 1
            cfg.output_dir = "/root/stock_price_prediction/tmp"
            main(cfg)

    def test_main_without_stock_code(self):
        with hydra.initialize(config_path="."):
            cfg = hydra.compose(config_name="config.yaml")
            cfg.output_dir = "/root/stock_price_prediction/tmp"
            with self.assertRaises(ValueError):
                main(cfg)

    def test_main_without_output_dir(self):
        with hydra.initialize(config_path="."):
            cfg = hydra.compose(config_name="config.yaml")
            cfg.output_dir = "hogehoge"
            with self.assertRaises(ValueError):
                main(cfg)

if __name__ == "__main__":
    unittest.main()
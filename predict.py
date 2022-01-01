from dataset import get_historical_data
import pandas as pd
import hydra
import os
import tensorflow as tf
import numpy as np


predictions = []
gts = []
is_win = []

def predict(code, model_dir, debug_date):
    df = get_historical_data(code)
    if debug_date is not None:
        df_gt = df.iloc[-debug_date]
        df = df.iloc[:-debug_date]
    df["Date"] = pd.to_datetime(df["Date"]) #TODO: this is just a WA
    # Test_X = df["Adj. Close"].to_numpy()[-5:].reshape(-1, 5, 1)
    Test_X = df["Adj. Close"].to_numpy()[-10:].reshape(-1, 10, 1)
    timestamp = df["Date"].to_numpy()[-1]

    model = tf.keras.models.load_model(os.path.join(model_dir))
    # import pdb;pdb.set_trace()
    prediction = model.predict(Test_X).reshape(-1)
    predictions.append(prediction)
    Test_X = Test_X.reshape(-1)
    if debug_date is not None:
        print(" ========== DEBUG ==========")
        print("Groundtruth tomorrow's data:")
        print(df_gt)
        gts.append(df_gt["Adj. Close"])
        # import pdb;pdb.set_trace()
        if len(predictions) >= 3 and np.sign(prediction[0] - gts[-3]) > 0:
            is_win.append(
                    df_gt["Adj. Close"] - gts[-3]
                )
        else:
            is_win.append(0)
    return prediction, Test_X, prediction[0] - Test_X[-1], timestamp

@hydra.main(config_name="config.yaml")
def main(cfg):
    if "debug" in cfg:
        for debug in range(5,100,1):
            cfg["debug"] = debug
            pred, today, delta, timestamp = predict(cfg.stock_code, cfg.eval.model_dir, cfg.debug)
            print("next 5 days:", pred)
            print("these 5 days:", today)
            print("delta:", delta)
            print("time:", timestamp)
            print(is_win)
            print(gts)
            print(sum(is_win) / len(is_win))
    else:
        # pred, today, delta, timestamp = predict(cfg.stock_code, cfg.eval.model_dir, None)
        pred, today, delta, timestamp = predict(cfg.stock_code, cfg.eval.model_dir, cfg.debug)
        print("next 5 days:", pred)
        print("these 5 days:", today)
        print("delta:", delta)
        print("time:", timestamp)

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    main()
from dataset import get_historical_data
import pandas as pd
import hydra
import os
import tensorflow as tf


def predict(code, model_dir):
    df = get_historical_data(code)
    df["Date"] = pd.to_datetime(df["Date"]) #TODO: this is just a WA
    Test_X = df["Adj. Close"].to_numpy()[-5:].reshape(-1, 5, 1)
    # import pdb;pdb.set_trace()
    timestamp = df["Date"].to_numpy()[-1]

    model = tf.keras.models.load_model(os.path.join(model_dir))
    prediction = model.predict(Test_X).reshape(-1)
    Test_X = Test_X.reshape(-1)
    return prediction, Test_X, prediction[0] - Test_X[-1], timestamp

@hydra.main(config_name="config.yaml")
def main(cfg):
    pred, today, delta, timestamp = predict(cfg.stock_code, cfg.eval.model_dir)
    print("next 5 days, these 5 days, delta, time:", pred, today, delta, timestamp)

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    main()
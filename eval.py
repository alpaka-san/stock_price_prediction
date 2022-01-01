import hydra
import os
from dataset import Dataset
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def plot_win_ratio(prediction, Test_Y, model_dir):
    ave = []
    Test_Y = Test_Y.reshape(-1)[::5]
    prediction = prediction.reshape(-1)[::5]
    for k in range(Test_Y.shape[0]-30):
        ave.append(
            np.sum(
                (np.squeeze(np.sign(Test_Y[1+k:21+k] - Test_Y[k:20+k]))
                ) * np.sign(prediction[1+k:21+k] - prediction[k:20+k])
            ) / np.prod(np.shape(prediction[k:20+k]))
        )
    plt.plot(np.array(ave)/2 + 1/2)
    plt.xlabel("Time")
    plt.ylabel("Win ratio of the Closing Price 20-days Mean value")
    os.makedirs(os.path.join(model_dir, "../png"), exist_ok=True)
    plt.savefig(os.path.join(model_dir, "../png/win_ratio.png"))
    plt.close()

def plot_revenue(prediction, Test_X, Test_Y, Test_Y_open, model_dir):
    pred = prediction.reshape(-1)[0::5]
    GT = Test_Y.reshape(-1)[0::5]
    ans = []
    stock_num = 1
    buy_trans_fee = 1.0006
    sell_trans_fee = 0.9994
    # import pdb;pdb.set_trace()
    for k in range(1, len(pred)-1):
        # if (pred[k] > 1.0*GT[k-1]) and (pred[k+1] > GT[k-1]):
        if (pred[k+1] > GT[k-1]):
            # ans.append(stock_num * (sell_trans_fee*GT[k] - buy_trans_fee*GT[k-1]))
            ans.append(stock_num * (sell_trans_fee*GT[k+1] - buy_trans_fee*GT[k-1]))
        else:
            ans.append(stock_num * (sell_trans_fee*Test_Y_open.reshape(-1)[0::5][k+1] - buy_trans_fee*GT[k-1]))
            # ans.append(0)
    plt.plot([sum(ans[:k])+GT[0] for k in range(len(ans))], label=f"model")
    # for k,s in enumerate(ans):
    #     # if not s == 0:
    #     # if not np.sign(GT[k] - GT[k-1]) == np.sign(pred[k] - pred[k-1]):
    #     if pred[k] > GT[k-1]:
    #         plt.scatter(k, pred[k], c="blue", linewidths=0.2, alpha=0.3)
    #     else:
    #         plt.scatter(k, pred[k], c="orange", linewidths=0.2, alpha=0.3)
    plt.plot(GT, label="GT (buy&hold)")
    plt.legend()
    os.makedirs(os.path.join(model_dir, "../png"), exist_ok=True)
    plt.savefig(os.path.join(model_dir, "../png/revenue.png"))
    plt.close()
    print("ANS:", ans)


def eval(code, date, model_dir):
    if date is None:
        # enter a logic here that date is set to today when data is not specified. 
        pass
    dataset = Dataset(code, date)
    model = tf.keras.models.load_model(os.path.join(model_dir))
    Test_X, Test_Y, timestamp = dataset.test_data
    Test_Y_open, _ = dataset.test_open_data
    prediction = model.predict(Test_X)
    plot_win_ratio(prediction, Test_Y, model_dir)
    plot_revenue(prediction, Test_X, Test_Y, Test_Y_open, model_dir)
    print("Test_X[-1]:", Test_X[-1])
    return prediction[-1], Test_Y[-2], Test_Y[-2] - prediction[-1], timestamp.to_numpy()[-1]

@hydra.main(config_name="config.yaml")
def main(cfg):
    pred, today, delta, timestamp = eval(cfg.stock_code, cfg.eval.date, cfg.eval.model_dir)
    print("prediction, today, delta, time:", pred, today, delta, timestamp)

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    main()

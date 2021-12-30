from models import Model
import os
from dataset import Dataset
import hydra
import tensorflow as tf
import matplotlib.pyplot as plt

def scheduler(epoch):
  
#   if epoch <= 150:
#     lrate = (10 ** -5) * (epoch / 150) 
  if epoch <= 100:
    lrate = (10 ** -5) * (epoch / 100) 
#   elif epoch <= 400:
  elif epoch <= 200:
    initial_lrate = (10 ** -5)
    k = 0.02
    # lrate = initial_lrate * math.exp(-k * (epoch - 150))
    lrate = initial_lrate * math.exp(-k * (epoch - 100))
  else:
    lrate = (10 ** -6)
  
  return lrate

def plot(output_dir, hist):
    history_dict = hist.history

    loss = history_dict["loss"]
    root_mean_squared_error = history_dict["root_mean_squared_error"]
    val_loss = history_dict["val_loss"]
    val_root_mean_squared_error = history_dict["val_root_mean_squared_error"]

    epochs = range(1, len(loss) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    
    fig.set_figheight(5)
    fig.set_figwidth(15)
    
    ax1.plot(epochs, loss, label = "Training Loss")
    ax1.plot(epochs, val_loss, label = "Validation Loss")
    ax1.set(xlabel = "Epochs", ylabel = "Loss")
    ax1.legend()
    
    ax2.plot(epochs, root_mean_squared_error, label = "Training Root Mean Squared Error")
    ax2.plot(epochs, val_root_mean_squared_error, label = "Validation Root Mean Squared Error")
    ax2.set(xlabel = "Epochs", ylabel = "Loss")
    ax2.legend()
    
    plt.savefig(os.path.join(output_dir, "loss.png"))

@hydra.main(config_name="config.yaml")
def main(cfg):
    if not os.path.exists(cfg.output_dir):
        raise ValueError(f"""
            Directory {cfg.output_dir} not found. Specify an appropriate output_dir.
        """)
    model = Model()
    dataset = Dataset()
    callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
    model.compile(
        optimizer = tf.keras.optimizers.Adam(),
        loss = 'mse',
        metrics = tf.keras.metrics.RootMeanSquaredError()
    )
    hist = model.fit(
        dataset.train_data[0],
        dataset.train_data[1],
        epochs = cfg.training.epochs,
        validation_data = (dataset.val_data[0], dataset.val_data[1]),
        callbacks=[callback]
    )
    model.save(os.path.join(cfg.output_dir, "my_model"))
    plot(cfg.output_dir, hist) # plot analysis

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    main()
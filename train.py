from models import Model


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


def main():
	model = Model()
    callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
	model.compile(optimizer = tf.keras.optimizers.Adam(), loss = 'mse', metrics = tf.keras.metrics.RootMeanSquaredError())
	hist = model.fit(AMZN_Train_X, AMZN_Train_Y, epochs = 100, validation_data = (AMZN_Val_X, AMZN_Val_Y), callbacks=[callback])
	model.save("my_model")
	plot(hist) # plot analysis

if "__name__" == "__main__":
    main()
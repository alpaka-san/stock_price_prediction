from PIL import Image


def plot_win_ratio(prediction):
		...
		Image.save("./png/win_ratio.png")

def plot_revenue(prediction):
		...
		Image.save("./png/revenue.png")

@hydra
def eval(code, date, model_dir):
		dataset = Dataset(code, date)
		model = keras.modesl.load_model(os.path.join(model_dir, "/path/to/model"))
		Test_X, Test_Y = dataset.test_data
		prediction = model.predict(Test_X)
		
		plot_win_ratio(prediction)
		plot_revenue(prediction)
		return prediction[-1], GT[-2], GT[-2] - prediction[-1]
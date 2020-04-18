import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def visualise_loss():
	model_name = "bvb_8"
	train_data_path = "./" + model_name + "_count_states_win_rate.txt"

	df = pd.read_csv(train_data_path, sep=",", header=None)
	df.columns = ["learning_rate", "mean_epsilon", "loss", "acc"]

	sns.lineplot(x=df.index, y='loss', data=df)
	plt.show()

	sns.lineplot(x=df.index, y='mean_epsilon', data=df)
	plt.show()



if __name__ == "__main__":
	visualise_loss()

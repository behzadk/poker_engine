import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def visualise_loss():
	model_name = "model_6/"
	train_data_path = "./model_checkpoints/checkpoint_" + model_name + "train_data.txt"

	df = pd.read_csv(train_data_path, sep=",", header=None)
	df.columns = ["count_states", "learning_rate", "mean_epsilon", "loss", "acc"]

	g = sns.lineplot(x=df.index, y='loss', data=df)
	g.set_yscale("log")
	g.set_ylim(None, 1)
	plt.show()

	sns.lineplot(x=df.index, y='mean_epsilon', data=df)
	plt.show()



if __name__ == "__main__":
	visualise_loss()

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from dataEncoder import *
import pandas as pd

def plot_data(df, n_dim):
	"""
	Warning : df must not contain the labels.
	"""
	pca = PCA(n_components=n_dim)
	X_pca = pca.fit_transform(df)

	if n_dim == 1:
		plt.plot(X_pca)
		plt.show()
	elif n_dim == 2:
		plt.scatter(X_pca[:,0], X_pca[:,1])
		plt.show()
	elif n_dim == 3:
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		ax.scatter(X_pca[:,0], X_pca[:,1], X_pca[:,2])
		plt.show()
	else:
		print("Dimension not supported for plotting")


if __name__ == "__main__":
	df = pd.read_csv("../HSTopdeck.csv").drop(columns=["card_type", "card_mark"]).set_index("name")
	de = DataEncoder()
	resulting_df = de.encode(df, 100).fillna(0)
	resulting_df.to_csv("test.csv")
	n_dim = 3
	plot_data(resulting_df, n_dim)

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
	elif n_dim == 2:
		plt.scatter(X_pca[:,0], X_pca[:,1])
	elif n_dim == 3:
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		ax.scatter(X_pca[:,0], X_pca[:,1], X_pca[:,2])
	else:
		print("Dimension not supported for plotting")
	plt.title(f"Cards in {n_dim} dimension after the encoding and a PCA")
	plt.show()


def plot_grades(grades) -> None:
	"""
	:param grades: a pandas Series containing the grades of the cards
	:return: plot the repartition of the grades
	"""
	n, bins, patches = plt.hist(grades, bins=10, density=False, cumulative=False)

	plt.title('Repartition of grades')
	plt.xlabel('Grades')
	plt.ylabel('Number of cards')

	plt.show()


if __name__ == "__main__":
	# Plotting label distribution
	df = pd.read_csv("../HSTopdeck.csv")["card_mark"]
	plot_grades(df)
	# Plotting points in space
	df = pd.read_csv("../HSTopdeck.csv").drop(columns=["card_type", "durability", "card_mark"]).set_index("name")
	de = DataEncoder()
	resulting_df = de.encode(df, 55).fillna(0)
	n_dim = 2
	plot_data(resulting_df, n_dim)

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import normalize, MinMaxScaler
import sys
sys.path.append('..')  # ajouter le dossier parent au PATH
from src.dataEncoder import *
from time import time
import matplotlib.pyplot as plt


class K_Means:

	def __init__(self, x, n_clusters):
		self.x = x
		self.model = KMeans(n_clusters=n_clusters)

	def fit(self):
		self.model.fit(self.x)

	def evaluate(self):
		y_pred = self.model.predict(self.x)
		return silhouette_score(self.x, y_pred)

	def pred(self, data):
		return self.model.predict(data)


if __name__ == "__main__":
	# read CSV
	df = pd.read_csv("../../HSTopdeck.csv", index_col=0)
	df = df.drop(columns=["card_type", "durability"])
	x, y = df.loc[:, ~df.columns.str.contains("card_mark")], df["card_mark"]
	data_encoder = DataEncoder()
	# encode and normalize data
	x_encoded = data_encoder.encode(x, 55).fillna(0)
	scaler = MinMaxScaler()
	cols_to_scale = ['mana', 'attack', 'health']
	x_encoded[cols_to_scale] = scaler.fit_transform(x_encoded[cols_to_scale])

	# Cross validating the number of clusters
	sse, slc = dict(), dict()
	random_seed = 42  # set to a fixed value if you want to reproduce
	for k in range(2, 50):
		kmeans = KMeans(n_clusters=k, max_iter=1000, n_init=10, random_state=random_seed).fit(x_encoded)
		clusters = kmeans.labels_
		sse[k] = kmeans.inertia_  # Inertia: Sum of distances of samples to their closest cluster center
		slc[k] = silhouette_score(x_encoded, clusters)

	plt.figure(figsize=(15, 10))
	plt.plot(list(sse.keys()), list(sse.values()))
	plt.xlabel("Number of cluster")
	plt.ylabel("SSE")
	plt.show()

	plt.figure(figsize=(15, 10))
	plt.plot(list(slc.keys()), list(slc.values()))
	plt.xlabel("Number of cluster")
	plt.ylabel("Silhouette score")
	plt.show()


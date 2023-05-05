from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import normalize
from src.dataEncoder import *
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
	df = pd.read_csv("../../HSTopdeck.csv", index_col=0)
	df = df.drop(columns=["card_type", "durability"])
	x, y = df.loc[:, ~df.columns.str.contains("card_mark")], df["card_mark"]
	data_encoder = DataEncoder()

	x_encoded = data_encoder.encode(x, 55).fillna(0)
	x_normalized = normalize(x_encoded)

	sse, slc = dict(), dict()
	for k in range(2, 20):
		# seed of 10 for reproducibility.
		kmeans = KMeans(n_clusters=k, max_iter=1000, random_state=10).fit(x_normalized)
		if k == 3:
			labels = kmeans.labels_
		clusters = kmeans.labels_
		sse[k] = kmeans.inertia_  # Inertia: Sum of distances of samples to their closest cluster center
		slc[k] = silhouette_score(x_normalized, clusters)

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


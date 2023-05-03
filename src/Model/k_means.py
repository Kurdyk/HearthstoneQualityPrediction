from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


class K_Means():

	def __init__(self,X,n_clusters):
		self.n_clusters = n_clusters
		self.X = X
		self.model = KMeans(n_clusters=self.n_clusters)

	def fit(self):
		self.model.fit(self.X)

	def evaluate(self):
		y_pred = self.model.predict(self.X)
		return silhouette_score(self.X, y_pred)

	def pred(self, data):
		return self.model.predict(data)

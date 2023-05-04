from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from src.dataEncoder import *
import pandas as pd
import matplotlib.pyplot as plt


class LinRegression:

	def __init__(self, x_train, y_train, x_test, y_test, alpha=0.1):
		self.x_train = x_train
		self.y_train = y_train
		self.x_test = x_test
		self.y_test = y_test
		self.model = LinearRegression()

	def fit(self):
		self.model.fit(self.x_train, self.y_train)

	def evaluate(self):
		y_pred = self.model.predict(self.x_test)
		return mean_squared_error(self.y_test, y_pred)

	def pred(self, data):
		return self.model.predict(data)


if __name__ == "__main__":
	df = pd.read_csv("../../HSTopdeck.csv", index_col=0)
	df = df.drop(columns=["card_type", "durability"])
	x, y = df.loc[:, ~df.columns.str.contains("card_mark")], df["card_mark"]
	scores = dict()
	for i in range(20, 100, 5):
		x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, shuffle=True)
		data_encoder = DataEncoder()

		x_train_encoded = data_encoder.encode(x_train, i).fillna(0)
		x_test_encoded = data_encoder.encode(x_test, i).fillna(0)

		x_train_normalized = normalize(x_train_encoded)
		x_test_normalized = normalize(x_test_encoded)

		lin_regressor = LinRegression(x_train_normalized, y_train, x_test_normalized, y_test)
		lin_regressor.fit()
		scores[i] = lin_regressor.evaluate()

	plt.figure(figsize=(30, 20))
	plt.plot(list(scores.keys()), list(scores.values()))
	plt.xlabel("# of LSA columns")
	plt.ylabel("MSE values")
	plt.show()



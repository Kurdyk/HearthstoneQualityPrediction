from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from ..dataEncoder import DataEncoder
import pandas as pd


class L1Regression:

	def __init__(self, x_train, y_train, x_test, y_test, alpha):
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
	df = df[df["card_type"] == "Minion"].drop(columns=["card_type", "durability"])
	x, y = df.loc[:, ~df.columns.str.contains("card_mark")], df["card_mark"]
	x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, shuffle=True, random_state=52)
	data_encoder = DataEncoder()

	x_train_encoded = data_encoder.encode(x_train, 40).fillna(0).drop(columns=["is_Shadow", "is_Nature", "is_Fire", "is_Arcane", "is_Frost", "is_Fel", "is_Holy"])
	x_test_encoded = data_encoder.encode(x_test, 40).fillna(0).drop(columns=["is_Shadow", "is_Nature", "is_Fire", "is_Arcane", "is_Frost", "is_Fel", "is_Holy"])

	x_train_normalized = normalize(x_train_encoded)
	x_test_normalized = normalize(x_test_encoded)

	# pca = PCA(n_components=10)
	# x_train_normalized = pca.fit_transform(x_train_normalized)
	# x_test_normalized = pca.fit_transform(x_test_normalized)

	l1_regressor = L1Regression(x_train_normalized, y_train, x_test_normalized, y_test, 0.1)
	l1_regressor.fit()
	print(l1_regressor.model.coef_)
	print(l1_regressor.evaluate())
	prediction = l1_regressor.pred(x_test_normalized)
	print(prediction)
	# for i in range(len(prediction)):
	# 	print(f"for {x_test_normalized[i]} score predicted: {prediction[i]}, expected {y_test[i]}")

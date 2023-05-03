from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from src.dataEncoder import *
import pandas as pd


class L1Regression:

	def __init__(self, x_train, y_train, x_test, y_test, alpha):
		self.x_train = x_train
		self.y_train = y_train
		self.x_test = x_test
		self.y_test = y_test
		self.model = Lasso(alpha=alpha)

	def fit(self):
		self.model.fit(self.x_train, self.y_train)

	def evaluate(self):
		y_pred = self.model.predict(self.x_test)
		return mean_squared_error(self.y_test, y_pred)

	def pred(self, data):
		return self.model.predict(data)


if __name__ == "__main__":
	df = pd.read_csv("../../HSTopdeck.csv", index_col=0).drop(columns="card_type")
	x, y = df.loc[:, ~df.columns.str.contains("card_mark")], df["card_mark"]
	x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, shuffle=True, random_state=42)
	data_encoder = DataEncoder(ClassEncoder(), MinionTypeEncoder(), RarityEncoder(),
							   SpellSchoolsEncoder(), TextEncoder())

	x_train_encoded = data_encoder.encode(x_train).fillna(0)
	x_test_encoded = data_encoder.encode(x_test).fillna(0)

	x_train_normalized = normalize(x_train_encoded)
	x_test_normalized = normalize(x_test_encoded)

	l1_regressor = L1Regression(x_train_normalized, y_train, x_test_normalized, y_test, 0.1)
	l1_regressor.fit()
	print(l1_regressor.evaluate())
	prediction = l1_regressor.pred(x_test_normalized)
	print(prediction)
	# for i in range(len(prediction)):
	# 	print(f"for {x_test_normalized[i]} score predicted: {prediction[i]}, expected {y_test[i]}")

from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error

class L1Regression():

	def __init__(self,X_train,y_train,X_test,y_test,alpha):
		self.X_train = X_train
		self.y_train = y_train
		self.X_test = X_test
		self.model = Lasso(alpha=alpha)

	def fit(self):
		self.model.fit(self.X_train, self.y_train)

	def evaluate(self):
		y_pred = self.model.predict(self.X_test)
		return mean_squared_error(y_test, y_pred)

	def pred(self,data):
		return self.model.predict(data)
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize,MinMaxScaler
from src.dataEncoder import *
from src.Model.regression import *
from src.Model.k_means import *
from src.dataVisualization import *
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

def print_best_dim_PCA(df):
	# Centrage et réduction des données
	X = df - np.mean(df, axis=0)
	X /= np.std(X, axis=0)

	# Calcul de la matrice de covariance
	cov_mat = np.cov(X.T)

	# Calcul des valeurs propres et vecteurs propres de la matrice de covariance
	eig_vals, eig_vecs = np.linalg.eig(cov_mat)

	# Tri des valeurs propres dans l'ordre décroissant
	sorted_idx = eig_vals.argsort()[::-1]
	eig_vals = eig_vals[sorted_idx]

	# Calcul de la variance expliquée pour chaque composante principale
	variance_ratio = eig_vals / np.sum(eig_vals)

	# Affichage de la variance cumulée expliquée en fonction du nombre de composantes principales
	cumulative_variance_ratio = np.cumsum(variance_ratio)
	print("Variance cumulée expliquée : ", cumulative_variance_ratio)

	# Recherche de la meilleure dimension à utiliser
	best_dim = np.argmax(cumulative_variance_ratio >= 0.95) + 1
	print("Meilleure dimension : ", best_dim)
	return best_dim

df = pd.read_csv("HSTopdeck.csv", index_col=0)

# À METTRE DANS L'ENCODING
df = df.replace('\xa0', '', regex=True)
df['card_text'] = df['card_text'].str.replace('\n', '')
df['card_text'] = df['card_text'].str.replace('"', '')
df['card_text'] = df['card_text'].str.replace(',', ' ')
df['card_text'] = df['card_text'].str.replace(':', ': ')


df = df[df["card_type"] == "Minion"].drop(columns=["card_type", "durability"])
x, y = df.loc[:, ~df.columns.str.contains("card_mark")], df["card_mark"]

# On scale les données numériques
scaler = MinMaxScaler()
cols_to_scale = ['mana', 'attack', 'health']
x[cols_to_scale] = scaler.fit_transform(x[cols_to_scale])

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, shuffle=True, random_state=52)

data_encoder = DataEncoder()
x_train_encoded = data_encoder.encode(x_train, 40).fillna(0).drop(columns=["is_Shadow", "is_Nature", "is_Fire", "is_Arcane", "is_Frost", "is_Fel", "is_Holy"])
x_test_encoded = data_encoder.encode(x_test, 40).fillna(0).drop(columns=["is_Shadow", "is_Nature", "is_Fire", "is_Arcane", "is_Frost", "is_Fel", "is_Holy"])

best_dim_pca = print_best_dim_PCA(x_train_encoded)

pca = PCA(n_components=best_dim_pca)
pca_x_train_encoded = pca.fit_transform(x_train_encoded)
pca_x_test_encoded = pca.fit_transform(x_test_encoded)

l1_regressor = L1Regression(pca_x_train_encoded, y_train, pca_x_test_encoded, y_test, 0.1)
l1_regressor.fit()
evaluation = l1_regressor.evaluate()

print(evaluation)

# Faire les prédictions sur les données d'entraînement
y_pred = l1_regressor.pred(pca_x_test_encoded.reshape(1,-1))

# Tracer les vraies valeurs par rapport aux prédictions
plt.scatter(y_test, y_pred)
plt.xlabel('Vraies valeurs')
plt.ylabel('Prédictions')
plt.title('Régression linéaire')
plt.show()

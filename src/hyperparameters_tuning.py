from Model.regression import *
from sklearn.model_selection import KFold
from mpl_toolkits.mplot3d import Axes3D
import random

import warnings
warnings.simplefilter(action='ignore', category=Warning)

"""
We're using a gridsearch on lsa and pca dim to find the best combination. We've to implement it ourselves because sklearn didn't provide the exact thing 
we want.
(We could also have used the variance explained but the grid search allows to test the effect of the combined LSA and PCA)
"""

best_avg_mse = 100000000
best_lsa_dim = -1
best_pca_dim = -1
resulting_graph = []
n_folds = 5

# Récupération et nettoyage du csv
df = pd.read_csv("../HSTopdeck.csv", index_col=0)
df = df.drop(columns=["card_type", "durability"])
x, y = df.loc[:, ~df.columns.str.contains("card_mark")], df["card_mark"]

# Séparation train/test
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.80, shuffle=True)

# Encodage des données
data_encoder = DataEncoder()
scaler = MinMaxScaler()
cols_to_scale = ['mana', 'attack', 'health']

# Gridsearch
for lsa_dim in range(0,101):
	rand = random.uniform(0,1)
	if rand <= 0.5: # To speed up gridsearch
		continue
	for pca_dim in range(1,35 + lsa_dim): # 35 + lsa_dim = number of cols of the dataset
		rand = random.uniform(0,1)
		if rand <= 0.5: # To speed up gridsearch
			continue

		print("---------------")
		print("lsa_dim : ",lsa_dim,", pca_dim : ",pca_dim)
		print("---------------")

		x_train_encoded = data_encoder.encode(df = x_train,n_dim_text = lsa_dim).fillna(0)
		x_test_encoded = data_encoder.encode(df = x_test,n_dim_text = lsa_dim).fillna(0)

		# Cross-validation
		kf = KFold(n_splits=n_folds)

		avg_mse = 0
		for train_index, test_index in kf.split(x_train_encoded):

			x_train_cv, x_test_cv = x_train_encoded.iloc[train_index], x_train_encoded.iloc[test_index]
			y_train_cv, y_test_cv = y_train.iloc[train_index], y_train.iloc[test_index]

			# We scaled and use PCA in the cross-validation to avoid dataleak
			x_train_cv[cols_to_scale] = scaler.fit_transform(x_train_cv[cols_to_scale])
			x_test_cv[cols_to_scale] = scaler.fit_transform(x_test_cv[cols_to_scale])

			pca = PCA(n_components=pca_dim)
			x_train_cv = pca.fit_transform(x_train_cv)
			x_test_cv = pca.fit_transform(x_test_cv)

			model = LinRegression(x_train_cv,y_train_cv,x_test_cv,y_test_cv)
			model.fit()

			avg_mse += model.evaluate()

		avg_mse = avg_mse/n_folds

		resulting_graph.append((lsa_dim,pca_dim,avg_mse))
		print("resulting_graph : ",resulting_graph)

		if avg_mse < best_avg_mse:
			best_avg_mse = avg_mse
			best_lsa_dim = lsa_dim
			best_pca_dim = pca_dim

print("---------------")
print("best_avg_mse : ",best_avg_mse)
print("best_lsa_dim : ",best_lsa_dim)
print("best_pca_dim : ",best_pca_dim)
print("---------------")

# We show the resulting graph
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

lsa_dim_list = []
pca_dim_list = []
mse_list = []

for (lsa_dim,pca_dim,mse) in resulting_graph:
	lsa_dim_list.append(lsa_dim)
	pca_dim_list.append(pca_dim)
	mse_list.append(mse)

ax.scatter(lsa_dim_list, pca_dim_list, mse_list)

ax.set_xlabel('lsa_dim')
ax.set_ylabel('pca_dim')
ax.set_zlabel('mean_squared_error')

plt.show()



from Model.regression import *

"""
We're using a gridsearch on lsa and pca dim to find the best combination.
"""

best_mse = 100000000
best_lsa_dim = -1
best_pca_dim = -1

# Récupération et nettoyage du csv
df = pd.read_csv("../HSTopdeck.csv", index_col=0)
df = df.drop(columns=["card_type", "durability"])
x, y = df.loc[:, ~df.columns.str.contains("card_mark")], df["card_mark"]
df = df.replace('\xa0', '', regex=True)
df['card_text'] = df['card_text'].str.replace('\n', '')
df['card_text'] = df['card_text'].str.replace('"', '')
df['card_text'] = df['card_text'].str.replace(',', ' ')
df['card_text'] = df['card_text'].str.replace(':', ' : ')
df['card_text'] = df['card_text'].str.replace('.', ' ')

# Séparation train/test
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, shuffle=True)

# Encodage des données
data_encoder = DataEncoder()
scaler = MinMaxScaler()
cols_to_scale = ['mana', 'attack', 'health']

for lsa_dim in range(1,5):
	for pca_dim in range(1,5):

		x_train_encoded = data_encoder.encode(x_train, lsa_dim).fillna(0)
		x_test_encoded = data_encoder.encode(x_test, lsa_dim).fillna(0)

		for i in range(0,2):
			if i == 0:
				# Normalisation des données
				x_train_normalized = normalize(x_train_encoded)
				x_test_normalized = normalize(x_test_encoded)

				# PCA
				

				# Création et entraînement du modèle
				lin_regressor = LinRegression(x_train_normalized, y_train, x_test_normalized, y_test)
				lin_regressor.fit()


			if i == 1:
				# Scaling des données
				x_train_encoded[cols_to_scale] = scaler.fit_transform(x_train_encoded[cols_to_scale])
				x_test_encoded[cols_to_scale] = scaler.fit_transform(x_test_encoded[cols_to_scale])	

				# PCA


				# Création et entraînement du modèle
				lin_regressor = LinRegression(x_train_encoded, y_train, x_test_encoded, y_test)
				lin_regressor.fit()

			# Évaluation du modèle
			mse_score = lin_regressor.evaluate()
			if mse_score < best_mse:
				best_mse = mse_score
				best_lsa_dim = lsa_dim
				best_pca_dim = pca_dim

				print("---------------")
				print("best_mse : ",best_mse)
				print("best_lsa_dim : ",best_lsa_dim)
				print("best_pca_dim : ",best_pca_dim)
				print("normalized(0)/scaled(1) : ",i)
				print("---------------")


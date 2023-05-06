from Model.regression import *
from dataVisualization import *

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
df.to_csv("clean_df.csv",index=False)

# Séparation train/test
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, shuffle=True)

# Encodage des données
data_encoder = DataEncoder()

lsa_dim = 55
x_train_encoded = data_encoder.encode(x_train, lsa_dim).fillna(0)
x_test_encoded = data_encoder.encode(x_test, lsa_dim).fillna(0)

# Normalisation des données / Scaling des données
"""
x_train_normalized = normalize(x_train_encoded)
x_test_normalized = normalize(x_test_encoded)
"""

scaler = MinMaxScaler()
cols_to_scale = ['mana', 'attack', 'health']
x_train_encoded[cols_to_scale] = scaler.fit_transform(x_train_encoded[cols_to_scale])
x_test_encoded[cols_to_scale] = scaler.fit_transform(x_test_encoded[cols_to_scale])


# Visualisation des données
plot_data(x_train_normalized,3)

# Création et entraînement du modèle
lin_regressor = LinRegression(x_train_encoded, y_train, x_test_encoded, y_test)
lin_regressor.fit()

# Évaluation du modèle
mse_score = lin_regressor.evaluate()
print(mse_score)
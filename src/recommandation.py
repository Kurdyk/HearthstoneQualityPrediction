import pandas as pd

from Model.k_means import *
from Model.regression import *
from numpy import nan
from ColumnEncoders.classEncoder import class_dict

keys = ["name", "mana", "card_text", "attack", "health", "durability", "class", "card_type", "rarity",
		"minion_type", "spell_school"]


def read_card_file(path: str) -> dict:
	"""
	:param path: path to the card file
	:return: a dict representing the card in the file
	"""
	result = {key: nan for key in keys}
	in_file = open(path, "r")
	lines = list()
	for line in in_file:
		lines.append(line.strip())
	in_file.close()
	for index in range(0, len(lines), 2):
		line = lines[index]
		next_line = lines[index + 1]
		for key in keys:
			if line.count(key) > 0:
				if key in {"minion_type", "class"}:
					result[key] = next_line.split(',')
				elif key in {"mana", "attack", "health", "durability"}:
					result[key] = int(next_line)
				else:
					result[key] = next_line
	print(f"Card read as follows: {result}")
	return result


if __name__ == "__main__":
	# Preparating models
	df = pd.read_csv("../HSTopdeck.csv", index_col=0)
	df = df.drop(columns=["card_type", "durability"])
	x, y = df.loc[:, ~df.columns.str.contains("card_mark")], df["card_mark"]

	data_encoder = DataEncoder()

	choice = input("Do you want to grade card (1) or recommand card remplacement (2)?\n")
	if choice == "1":  # card gradding
		x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, shuffle=True)
		print("Encoding data to train models")
		x_train_encoded = data_encoder.encode(x_train, 55).fillna(0)
		x_test_encoded = data_encoder.encode(x_test, 55).fillna(0)

		x_train_normalized = normalize(x_train_encoded)
		x_test_normalized = normalize(x_test_encoded)
		print("Done")
		print("Fitting model")
		lin_regressor = LinRegression(x_train_normalized, y_train, x_test_normalized, y_test)
		lin_regressor.fit()
		print("Done")
		# Asking for cards details
		while True:
			try:
				card = read_card_file(input("Path to card in a file ?\n"))
				card_df = pd.DataFrame(card, columns=keys).drop(columns=["card_type", "durability"]).set_index("name")
				if card["name"] in x_train.index:
					print(f"This card was present in the training set with a grade of {y_train.loc[y_train.index == card['name']][0]}")
					choice_bis = input("Do you want to retrain the model without it ? yes/no\n")
					if choice_bis == "yes":
						df_bis = df.drop(card["name"])
						x, y = df.loc[:, ~df.columns.str.contains("card_mark")], df["card_mark"]
						x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, shuffle=True)
						print("Encoding data to train the model")
						x_train_encoded = data_encoder.encode(x_train, 55).fillna(0)
						x_test_encoded = data_encoder.encode(x_test, 55).fillna(0)

						x_train_normalized = normalize(x_train_encoded)
						x_test_normalized = normalize(x_test_encoded)
						print("Done")
						lin_regressor_bis = LinRegression(x_train_normalized, y_train, x_test_normalized, y_test)
						print("Fitting model")
						lin_regressor_bis.fit()
						print("Done")
						print("Encoding new card")
						encoding_df = x_test._append(card_df)
						with_new_card = normalize(data_encoder.encode(encoding_df, 55).fillna(0))
						new_card_encoded = with_new_card[-1]
						print("Done")
						print(f"This card is graded {lin_regressor_bis.pred([new_card_encoded])[0]} by the retrained model")

				else:
					print("Encoding new card")
					encoding_df = x_test._append(card_df)
					with_new_card = normalize(data_encoder.encode(encoding_df, 55).fillna(0))
					new_card_encoded = with_new_card[-1]
					print("Done")
					print(f"This card is graded {lin_regressor.pred([new_card_encoded])[0]} by the model")

			except FileNotFoundError:
				print("not found")

	elif choice == "2":  # card recommandation
		print("Applying Kmeans and encoding cards")
		x_normalized = normalize(data_encoder.encode(x, 55).fillna(0))
		k_means = K_Means(x_normalized, 6)
		k_means.fit()
		labels = k_means.model.labels_
		print("Done")
		while True:
			try:
				card = read_card_file(input("Path to card in a file ?\n"))
				try:
					index = list(x.index).index(card["name"])
					cluster_index = k_means.pred([x_normalized[index]])
				except ValueError:
					card_df = pd.DataFrame(card, columns=keys).drop(columns=["card_type", "durability"]).set_index("name")
					encoding_df = x._append(card_df)
					print("Encoding new card")
					with_new_card = normalize(data_encoder.encode(encoding_df, 55).fillna(0))
					print("Done")
					new_card_encoded = with_new_card[-1]
					cluster_index = k_means.pred([new_card_encoded])

				cluster = df[labels == cluster_index]
				# find the index labels of all rows with the maximum rating with the right class
				compatible = pd.DataFrame(columns=keys).set_index("name")
				for index, elt in df.iterrows():
					classes = [hero_class for hero_class in class_dict if elt["class"].count(hero_class) > 0]
					for hero_class in classes:
						if hero_class in card["class"] + ["Neutral"] and elt["mana"] == card["mana"]:
							compatible = compatible._append(elt)
							continue
				try:
					n_remplacement = int(input("How many recommandations do you want per card ?\n"))
				except ValueError:
					print("Please give a valid number")
					continue
				best_match = compatible.nlargest(n_remplacement, "card_mark")
				for index, card_suggestion in best_match.iterrows():
					print(f"You can replace this card with {index} graded {card_suggestion['card_mark']}, you can see it"
						  f"at : https://www.hearthstonetopdecks.com/cards/{index}/")

			except FileNotFoundError:
				print("not found")
	else:
		print("1 or 2 as input, bye")

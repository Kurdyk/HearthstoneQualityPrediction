from Model.k_means import *
from Model.regression import *
from numpy import nan

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
	print(result)
	return result


if __name__ == "__main__":
	# Preparating models
	df = pd.read_csv("../HSTopdeck.csv", index_col=0)
	df = df.drop(columns=["card_type", "durability"])
	x, y = df.loc[:, ~df.columns.str.contains("card_mark")], df["card_mark"]
	data_encoder = DataEncoder()
	x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, shuffle=True)
	print("Encoding data to train models")
	x_train_encoded = data_encoder.encode(x_train, 55).fillna(0)
	x_test_encoded = data_encoder.encode(x_test, 55).fillna(0)

	x_train_normalized = normalize(x_train_encoded)
	x_test_normalized = normalize(x_test_encoded)
	print("Done")
	print("Applying Kmeans")
	k_means = K_Means(x_train_encoded, 6)
	k_means.fit()
	print("Done")
	lin_regressor = LinRegression(x_train_normalized, y_train, x_test_normalized, y_test)
	print("Fitting model")
	lin_regressor.fit()
	print("Done")
	# Asking for cards details
	while True:
		try:
			card = read_card_file(input("Path to card in a file ?\n"))
			card_df = pd.DataFrame(card, columns=keys).drop(columns=["card_type", "durability"]).set_index("name")
			print(card_df)
			encoding_df = x_test.append(card_df)
			print(encoding_df)
			with_new_card = normalize(data_encoder.encode(encoding_df, 55).fillna(0))
			new_card_encoded = with_new_card[-1]
			print(f"This card is graded {lin_regressor.pred([new_card_encoded])} by the model") # todo remove the wanted card from the training set to avoid leaks
			print(k_means.pred([new_card_encoded]))
			# todo: recommand the best cards in the same cluster with respect to class
		except FileNotFoundError:
			print("not found")

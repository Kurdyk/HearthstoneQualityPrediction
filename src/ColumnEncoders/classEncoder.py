import pandas as pd

class_dict = {"Druid": 0b1,
			  "Hunter": 0b10,
			  "Paladin": 0b100,
			  "Mage": 0b1000,
			  "Warrior": 0b10000,
			  "Shaman": 0b100000,
			  "Priest": 0b1000000,
			  "Demon Hunter": 0b10000000,
			  "Neutral": 0b100000000,
			  "Rogue": 0b1000000000,
			  "Warlock": 0b10000000000,
			  "Death Knight": 0b100000000000}


class ClassEncoder:

	def encode_class(self, class_list: list) -> dict:
		"""
		:param class_list: the list of the classes of the card
		:return: an hexa encoding of this list in a dict
		"""
		bit_dict_hexa = {"Class_hex0": 0, "Class_hex1": 0, "Class_hex2": 0}
		binary = 0b0
		for card_class in class_list:
			binary += class_dict[card_class]
		hexa = hex(binary)
		if len(hexa) < 5:  # fill the blanks
			value = hexa[2:]
			hexa = hexa[:2] + "0" * (3 - len(value)) + value
		for i in range(3):  # read the hexa encoding
			try:
				bit_dict_hexa["Class_hex" + str(i)] = (int('0x' + hexa[2 + i], base=16))
			except IndexError:
				pass
		return bit_dict_hexa

	def encode_class_col(self, dataframe: pd.DataFrame) -> pd.DataFrame:
		"""
		:param dataframe: A dataframe with a class column to encode using the hexa encoding
		:return: the dataframe with the class column removed and remplaced by its encoding
		"""
		def parse_list(list_as_str: str) -> list:
			"""
			:param list_as_str: List of classes of a card as a string (since datdframes save them as strings)
			:return: a python list with the classes
			"""
			return [hero_class for hero_class in class_dict if list_as_str.count(hero_class) > 0]

		class_encoding_df = pd.DataFrame(columns=["name", "Class_hex0", "Class_hex1", "Class_hex2"]).set_index("name")
		for index, row in dataframe["class"].items():
			name, classes = index, row
			class_list = parse_list(classes)
			encoding = self.encode_class(class_list)
			encoding["name"] = name
			encoding = {k: [v] for k, v in encoding.items()}
			tmp = pd.DataFrame(encoding, columns=["name", "Class_hex0", "Class_hex1", "Class_hex2"]).set_index("name")
			class_encoding_df = class_encoding_df._append(tmp)

		tmp = dataframe.join(class_encoding_df, on="name")
		tmp = tmp.drop(columns="class")
		tmp = tmp.loc[:, ~tmp.columns.str.contains('^Unnamed')]
		return tmp

	def encode_class_col_one_hot(self, dataframe: pd.DataFrame) -> pd.DataFrame:
		"""
		:param dataframe: A dataframe with a class column to encode using the one hot encoding
		:return: the dataframe with the class column removed and remplaced by its encoding
		"""
		def parse_list(list_as_str: str) -> list:
			"""
			:param list_as_str: List of classes of a card as a string (since datdframes save them as strings)
			:return: a python list with the classes
			"""
			return [hero_class for hero_class in class_dict if list_as_str.count(hero_class) > 0]

		new_col = [f"is_{class_name}" for class_name in class_dict]
		all_col = ["name"] + new_col
		class_encoding_df = pd.DataFrame(columns=all_col).set_index("name")

		for index, row in dataframe["class"].items():
			name, classes = index, row
			class_list = parse_list(classes)
			encoding = {f"is_{class_name}": 1 if class_name in class_list else 0 for class_name in class_dict}
			encoding["name"] = name
			encoding = {k: [v] for k, v in encoding.items()}
			tmp = pd.DataFrame(encoding, columns=all_col).set_index("name")
			class_encoding_df = class_encoding_df._append(tmp)

		tmp = dataframe.join(class_encoding_df, on="name")
		tmp = tmp.drop(columns="class")
		tmp = tmp.loc[:, ~tmp.columns.str.contains('^Unnamed')]
		return tmp


if __name__ == "__main__":
	df = pd.read_csv("HSTopdeck.csv").set_index("name")
	ce = ClassEncoder()
	ce.encode_class_col_one_hot(df).to_csv("test_class.csv")


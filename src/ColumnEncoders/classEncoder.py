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

	def __init__(self, dataframe: pd.DataFrame):
		self.dataframe = dataframe

	def encode_class(self, class_list: list):
		bit_dict_hexa = {"Class_hex0": 0, "Class_hex1": 0, "Class_hex2": 0}
		binary = 0b0
		for card_class in class_list:
			binary += class_dict[card_class]
		hexa = hex(binary)
		if len(hexa) < 5:
			value = hexa[2:]
			hexa = hexa[:2] + "0" * (3 - len(value)) + value
		for i in range(3):
			try:
				bit_dict_hexa["Class_hex" + str(i)] = (int('0x' + hexa[2 + i], base=16))
			except IndexError:
				pass
		return bit_dict_hexa

	def encode_class_col(self):

		def parse_list(list_as_str: str):
			return [hero_class for hero_class in class_dict if list_as_str.count(hero_class) > 0]

		class_encoding_df = pd.DataFrame(columns=["name", "Class_hex0", "Class_hex1", "Class_hex2"]).set_index("name")
		for index, row in self.dataframe[["name", "class"]].iterrows():
			name, classes = row["name"], row["class"]
			class_list = parse_list(classes)
			encoding = self.encode_class(class_list)
			encoding["name"] = name
			encoding = {k: [v] for k, v in encoding.items()}
			tmp = pd.DataFrame(encoding, columns=["name", "Class_hex0", "Class_hex1", "Class_hex2"]).set_index("name")
			class_encoding_df = class_encoding_df.append(tmp)

		tmp = self.dataframe.join(class_encoding_df, on="name")
		tmp = tmp.drop(columns="class")
		tmp = tmp.loc[:, ~tmp.columns.str.contains('^Unnamed')]
		return tmp


if __name__ == "__main__":
	df = pd.read_csv("../../HSTopdeck.csv")
	ce = ClassEncoder(df)
	ce.encode_class_col().to_csv("test_class.csv")


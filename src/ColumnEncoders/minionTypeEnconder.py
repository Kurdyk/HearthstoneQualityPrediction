import pandas as pd

type_dict = {"Amalgam": 0b1,
			 "Beast": 0b10,
			 "Demon": 0b100,
			 "Dragon": 0b1000,
			 "Elemental": 0b10000,
			 "Mech": 0b100000,
			 "Murloc": 0b1000000,
			 "Naga": 0b10000000,
			 "Pirate": 0b100000000,
			 "Quilboar": 0b1000000000,
			 "Totem": 0b10000000000,
			 "Undead": 0b100000000000}


class MinionTypeEncoder:

	def __init__(self, dataframe: pd.DataFrame):
		self.dataframe = dataframe

	def encode_class_col(self):

		def parse_list(list_as_str: str):
			return [minion_type for minion_type in type_dict if list_as_str.count(minion_type) > 0]

		def encode_type(type_list: list):
			bit_dict_hexa = {"Type_hex0": 0, "Type_hex1": 0, "Type_hex2": 0}
			binary = 0b0
			for card_type in type_list:
				binary += type_dict[card_type]
			hexa = hex(binary)
			if len(hexa) < 5:
				value = hexa[2:]
				hexa = hexa[:2] + "0" * (3 - len(value)) + value
			for i in range(3):
				try:
					bit_dict_hexa["Type_hex" + str(i)] = (int('0x' + hexa[2 + i], base=16))
				except IndexError:
					pass
			return bit_dict_hexa

		type_encoding_df = pd.DataFrame(columns=["name", "Type_hex0", "Type_hex1", "Type_hex2"]).set_index("name")
		for index, row in self.dataframe[["name", "minion_type"]].iterrows():
			name, types = row["name"], row["minion_type"]
			if self.dataframe.isnull().iloc[index]["minion_type"]:
				encoding = {"Type_hex0": 0, "Type_hex1": 0, "Type_hex2": 0}
			else:
				type_list = parse_list(types)
				encoding = encode_type(type_list)
			encoding["name"] = name
			encoding = {k: [v] for k, v in encoding.items()}
			tmp = pd.DataFrame(encoding, columns=["name", "Type_hex0", "Type_hex1", "Type_hex2"]).set_index("name")
			type_encoding_df = type_encoding_df.append(tmp)

		tmp = self.dataframe.join(type_encoding_df, on="name")
		tmp = tmp.drop(columns="minion_type")
		tmp = tmp.loc[:, ~tmp.columns.str.contains('^Unnamed')]
		return tmp


if __name__ == "__main__":
	df = pd.read_csv("test_class.csv")
	ce = MinionTypeEncoder(df)
	ce.encode_class_col().to_csv("test_type.csv")

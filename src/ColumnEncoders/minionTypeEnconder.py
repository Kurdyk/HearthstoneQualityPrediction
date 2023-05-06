import pandas as pd
import numpy as np

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

	def encode_type_col(self, dataframe: pd.DataFrame) -> pd.DataFrame:
		"""
		:param dataframe: A dataframe with a minion_type column to encode using the hexa encoding
		:return: the dataframe with the minion_type column removed and remplace by its encoding
		"""
		def parse_list(list_as_str: str) -> list:
			"""
			:param list_as_str: List of minion_types of a card as a string (since datdframes save them as strings)
			:return: a python list with the minion_types
			"""
			return [minion_type for minion_type in type_dict if list_as_str.count(minion_type) > 0]

		def encode_type(type_list: list):
			"""
			:param type_list: the list of the types of the card
			:return: an hexa encoding of this list in a dict
			"""
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
		for index, row in dataframe["minion_type"].items():
			name, types = index, row
			if type(types) == float:
				encoding = {"Type_hex0": 0, "Type_hex1": 0, "Type_hex2": 0}
			else:
				encoding = encode_type(parse_list(types))
			encoding["name"] = name
			encoding = {k: [v] for k, v in encoding.items()}
			tmp = pd.DataFrame(encoding, columns=["name", "Type_hex0", "Type_hex1", "Type_hex2"]).set_index("name")
			type_encoding_df = type_encoding_df._append(tmp)

		tmp = dataframe.join(type_encoding_df, on="name")
		tmp = tmp.drop(columns="minion_type")
		tmp = tmp.loc[:, ~tmp.columns.str.contains('^Unnamed')]
		return tmp

	def encode_type_col_one_hot(self, dataframe: pd.DataFrame):
		"""
		:param dataframe: A dataframe with a class column to encode using the one hot encoding
		:return: the dataframe with the class column removed and remplaced by its encoding
		"""
		def parse_list(list_as_str: str) -> list:
			"""
			:param list_as_str: List of minion_types of a card as a string (since datdframes save them as strings)
			:return: a python list with the minion_types
			"""
			return [minion_type for minion_type in type_dict if list_as_str.count(minion_type) > 0]

		new_col = [f"is_{type_name}" for type_name in type_dict]
		all_col = ["name"] + new_col
		type_encoding_df = pd.DataFrame(columns=all_col).set_index("name")

		for index, row in dataframe["minion_type"].items():
			name, types = index, row
			if type(types) == float:  # nan -> card without a minion_type
				encoding = {f"is_{minion_type}": 0 for minion_type in type_dict}
			else:
				encoding = {f"is_{minion_type}": 1 if minion_type in types else 0 for minion_type in type_dict}
			encoding["name"] = name
			encoding = {k: [v] for k, v in encoding.items()}
			tmp = pd.DataFrame(encoding, columns=all_col).set_index("name")
			type_encoding_df = type_encoding_df._append(tmp)

		tmp = dataframe.join(type_encoding_df, on="name")
		tmp = tmp.drop(columns="minion_type")
		tmp = tmp.loc[:, ~tmp.columns.str.contains('^Unnamed')]
		return tmp


if __name__ == "__main__":
	df = pd.read_csv("../../HSTopdeck.csv")
	ce = MinionTypeEncoder()
	ce.encode_type_col_one_hot(df).to_csv("test_type.csv")

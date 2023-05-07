import pandas as pd

rarities = {"Common", "Rare", "Epic", "Legendary"}


class RarityEncoder:

	def encode_rarity_col(self, dataframe: pd.DataFrame) -> pd.DataFrame:
		"""
		:param dataframe: A dataframe with a rarity column to encode
		:return: The dataframe with the rarity column removed and remplaced by its encoding, 1 for legendary, 0 for others
		"""
		all_col = ["name", "is_lengendary"]

		rarity_encoding_df = pd.DataFrame(columns=all_col).set_index("name")
		for index, row in dataframe["rarity"].items():
			name, rarity = index, row
			encoding = {"name": name, "is_legendary": 0}
			encoding = {k: v for k, v in encoding.items()}
			tmp = pd.DataFrame(encoding, columns=all_col, index=["name"]).set_index("name")
			tmp.iloc[0]["is_lengendary"] = 1 if rarity == "Legendary" else 0
			rarity_encoding_df = rarity_encoding_df._append(tmp)

		tmp = dataframe.join(rarity_encoding_df, on="name")
		tmp = tmp.drop(columns="rarity")
		tmp = tmp.loc[:, ~tmp.columns.str.contains('^Unnamed')]
		return tmp


if __name__ == "__main__":
	df = pd.read_csv("HSTopdeck.csv").set_index("name")
	ce = RarityEncoder()
	ce.encode_rarity_col(df).to_csv("test_rarity.csv")

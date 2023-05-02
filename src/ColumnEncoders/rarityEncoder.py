import pandas as pd

rarities = {"Common", "Rare", "Epic", "Legendary"}


class RarityEncoder:

	def __init__(self, dataframe: pd.DataFrame):
		self.dataframe = dataframe

	def encode_rarity_col(self):
		all_col = ["name", "is_lengendary"]

		rarity_encoding_df = pd.DataFrame(columns=all_col).set_index("name")
		for index, row in self.dataframe[["name", "rarity"]].iterrows():
			name, rarity = row["name"], row["rarity"]
			encoding = {"name": name, "is_legendary": 0}
			encoding = {k: v for k, v in encoding.items()}
			tmp = pd.DataFrame(encoding, columns=all_col, index=["name"]).set_index("name")
			tmp.iloc[0]["is_lengendary"] = 1 if rarity == "Legendary" else 0
			rarity_encoding_df = rarity_encoding_df.append(tmp)

		tmp = self.dataframe.join(rarity_encoding_df, on="name")
		tmp = tmp.drop(columns="rarity")
		tmp = tmp.loc[:, ~tmp.columns.str.contains('^Unnamed')]
		return tmp


if __name__ == "__main__":
	df = pd.read_csv("test_type.csv")
	ce = RarityEncoder(df)
	ce.encode_rarity_col().to_csv("test_rarity.csv")

import pandas as pd
import numpy as np

schools = {"Arcane", "Fel", "Fire", "Frost", "Holy", "Nature", "Shadow"}


class SpellSchoolsEncoder:

	def encode_spell_school_col(self, dataframe: pd.DataFrame) -> pd.DataFrame:
		"""
		:param dataframe: A dataframe with a rarity column to encode using a one hot encoding
		:return: The dataframe with the rarity column removed and remplaced by its encodings
		"""
		all_col = ["name"] + [f"is_{school}" for school in schools]
		school_encoding_df = pd.DataFrame(columns=all_col).set_index("name")
		for index, row in dataframe["spell_school"].items():
			name, current_school = index, row
			if current_school == np.nan:
				encoding = {f"is_{school}": 0 for school in schools}
			else:
				encoding = {f"is_{school}": 1 if current_school == school else 0 for school in schools}
			encoding["name"] = name
			encoding = {k: [v] for k, v in encoding.items()}
			tmp = pd.DataFrame(encoding, columns=all_col).set_index("name")
			school_encoding_df = school_encoding_df._append(tmp)

		tmp = dataframe.join(school_encoding_df, on="name")
		tmp = tmp.drop(columns="spell_school")
		tmp = tmp.loc[:, ~tmp.columns.str.contains('^Unnamed')]
		return tmp


if __name__ == "__main__":
	df = pd.read_csv("HSTopdeck.csv")
	ce = SpellSchoolsEncoder()
	ce.encode_spell_school_col(df).to_csv("test_school.csv")

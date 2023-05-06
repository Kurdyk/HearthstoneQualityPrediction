from src.ColumnEncoders.classEncoder import ClassEncoder
from src.ColumnEncoders.minionTypeEnconder import MinionTypeEncoder
from src.ColumnEncoders.rarityEncoder import RarityEncoder
from src.ColumnEncoders.spellSchoolEncoder import SpellSchoolsEncoder
from src.ColumnEncoders.textEncoder import TextEncoder
import pandas as pd


class DataEncoder:
	def __init__(self, class_encoder: ClassEncoder = ClassEncoder(),
				 minion_type_encoder: MinionTypeEncoder = MinionTypeEncoder(),
				 rarity_encoder: RarityEncoder = RarityEncoder(),
				 spell_school_encoder: SpellSchoolsEncoder = SpellSchoolsEncoder(),
				 text_encoder: TextEncoder = TextEncoder()):
		self.class_encoder = class_encoder
		self.minion_type_encoder = minion_type_encoder
		self.rarity_encoder = rarity_encoder
		self.spell_school_encoder = spell_school_encoder
		self.text_encoder = text_encoder

	def encode(self, df: pd.DataFrame, n_dim_text: int = 20, hexa=False):
		if hexa:
			class_encoded = self.class_encoder.encode_class_col(df)
			type_encoded = self.minion_type_encoder.encode_type_col(class_encoded)
		else:
			class_encoded = self.class_encoder.encode_class_col_one_hot(df)
			type_encoded = self.minion_type_encoder.encode_type_col_one_hot(class_encoded)
		rarity_encoded = self.rarity_encoder.encode_rarity_col(type_encoded)
		spell_school_encoded = self.spell_school_encoder.encode_spell_school_col(rarity_encoded)

		text_col = list(df["card_text"])
		text_encoded = self.text_encoder.encode(text_col, n_dim_text)
		tmp_dict = dict()
		for i in range(len(text_col)):
			card_name = df.index[i]
			tmp_dict[card_name] = text_encoded[i]
		tmp_dict = {k: v for k, v in tmp_dict.items()}
		test_df = pd.DataFrame(tmp_dict).transpose()
		test_df.columns = [f"lsa_{i}" for i in range(n_dim_text)]
		tmp_df = spell_school_encoded.join(test_df, on="name")
		text_encoded = tmp_df.drop(columns="card_text")
		return text_encoded


if __name__ == "__main__":
	df = pd.read_csv("../HSTopdeck.csv", index_col=0)
	data_encoder = DataEncoder(ClassEncoder(), MinionTypeEncoder(), RarityEncoder(),
							   SpellSchoolsEncoder(), TextEncoder())

	data_encoder.encode(df, 20).to_csv("encoded_data.csv")

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

"""
To do:
- On peut test de faire de la cross validation pour déterminer le bon n_dim
"""


class DataEncoder:
	def __init__(self, df: pd.DataFrame):
		self.df = df

	def encode(self,corpus,n_dim):
		"""
		Use TF-IDF to vectorize a corpus (string array). Then it used LSA on the TF-IDF matrix to reduce the dimensionality to n_dim.
		Warning : Non-alphanumeric characters at the beginning of a word (except space) are taken into account in the tokenization. Otherwise we 
		could not take into account word like "+X/+X"
		"""
		
		# Creating the columns using TF-IDF
		vectorizer = TfidfVectorizer(token_pattern=r"(?u)[\w+-/]+\b") # We've changed the token pattern because the default one ignore the special char like +,-,etc.
																	  # Moreover it also ignores word of length < 2 which reject all the number we can have in our text.
		tf_idf_matrix = vectorizer.fit_transform(corpus)
		
		# Applying LSA on the TF-IDF matrix
		lsa_model = TruncatedSVD(n_components=n_dim)
		lsa_matrix = lsa_model.fit_transform(tf_idf_matrix)

		return lsa_matrix


if __name__ == "__main__":
	df = pd.read_csv("../OldData/hearthstone.csv")
	de = DataEncoder(df)

	test_corpus = ["Discover a minion. Give it +1/+1.",
		"Enrage: +2 Attack.",
		"Deathrattle: Summon a random friendly Beast that died this game.",
		"At the end of your turn, eat a random enemy minion and gain its stats.",
		"Taunt. Deathrattle: Deal 2 damage to ALL characters.",
		"Discover a 6-Cost minion. Summon it with Taunt and Divine Shield.",
		"Spell Damage +2 Deathrattle:The next minion you draw inherits these powers."]
	resulting_df = de.encode(test_corpus,3)

	print(resulting_df)

	


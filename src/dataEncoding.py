import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

class DataEncoder:
	def __init__(self,df: pd.DataFrame):
		self.df = df


	def encode(self,corpus):
		"""
		Use TF-IDF to vectorize a corpus (string array). Return the dataframe obtained.
		Warning : Non-alphanumeric characters at the beginning of a word (except space) are taken into account in the tokenization. Otherwise we 
		could not take into account word like "+X/+X"
		"""
		
		# Creating the columns using TF-IDF
		vectorizer = TfidfVectorizer(token_pattern=r"(?u)[\w+-/]+\b") # We've changed the token pattern because the default one ignore the special char like +,-,etc.
																	  # Moreover it also ignores word of length < 2 which reject all the number we can have in our text.
		X = vectorizer.fit_transform(corpus)
		vector = vectorizer.get_feature_names_out()
		df = pd.DataFrame(columns=vector)

		# The columns are filled with data from the corpus
		for text in corpus:
			row = dict()
			for column in vector:
				row[column] = int(column in str.lower(text))
			df = df._append(row, ignore_index=True)

		return df

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
	resulting_df = de.encode(test_corpus)

	print(resulting_df)

	


import pandas as pd
import re


class DataEncoder:
	def __init__(self, df: pd.DataFrame):
		self.df = df

	def tokenization(self, documents):
		"""
		Use TF-IDF to encode the text and modifie the dataframe accordingly
		"""
		documents = self.df

		# Compute TF and IDF
		tf_array = list()  # array of dict, each dict represent a document and contain the tf of each words in the document
		idf_dict = dict()  # dict of words, each words is associated with the number of document which contain this word

		for document in documents:
			tf = dict()

			delimited_document = re.split(' | , | ; | :', document)

			for word in delimited_document:
				# idf update
				if word in idf_dict:
					idf_dict[word] += 1
				else:
					idf_dict[word] = 0

				# tf update
				if word in tf:
					tf[word] += 1
				else:
					tf[word] = 0

			for word in tf:
				tf[word] /= len(tf)

			tf_array.append(tf)



if __name__ == "__main__":
	df = pd.read_csv("../OldData/hearthstone.csv")
	de = DataEncoder(df)
	de.encode_class_col(["Rogue", "Demon Hunter"])


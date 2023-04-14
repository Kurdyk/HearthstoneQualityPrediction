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

	def encode_class_col(self, c):

		class_dict = {"Druid": 0b1, "Hunter": 0b10, "Paladin": 0b100, "Mage": 0b1000, "Warrior": 0b10000,
					  "Shaman": 0b10000, "Priest": 0b100000, "Demon Hunter": 0b1000000, "Neutral": 0b1000000,
					  "Rogue": 0b10000000, "Warlock": 0b100000000, "Death Knight": 0b1000000000}

		def encode_class(class_list: list):
			bit_list_hexa = list()
			binary = 0b0
			for card_class in class_list:
				binary += class_dict[card_class]
			hexa = hex(binary)
			for i in range(3):
				try:
					bit_list_hexa.append(int(hexa[2 + i]))
				except IndexError:
					bit_list_hexa.append(0)
			return bit_list_hexa

		print(encode_class(c))




if __name__ == "__main__":
	df = pd.read_csv("../OldData/hearthstone.csv")
	de = DataEncoder(df)
	de.encode_class_col(["Druid", "Paladin"])


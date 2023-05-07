import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup

base_url = "https://www.hearthstonetopdecks.com/cards/page/"
keys = ["name", "mana", "card_text", "attack", "health", "durability", "class", "card_type", "rarity", "card_mark",
		"minion_type", "spell_school"]


def parse_card_page(url, card_name) -> pd.DataFrame:
	"""
	:param url: page url of the card
	:param card_name: the name of the card
	:return: a dataframe containing the information of the card
	"""
	response = requests.get(url)
	soup = BeautifulSoup(response.text, "html.parser")
	card_info = soup.find("div", {"class": "col-md-14"})

	card_dict = {key: np.nan for key in keys}
	card_dict["name"] = card_name
	card_text = card_info.findNext("div", {"class": "card-content"}).findNext('p').findNext('p')
	card_dict["card_text"] = card_text.get_text()
	list_detail = card_info.findNext("ul")
	for li in list_detail.find_all("li"):
		li = li.get_text()
		if li.count("Mana Cost:") > 0:
			card_dict["mana"] = int(li[-2:])
		elif li.count("Attack:") > 0:
			card_dict["attack"] = int(li[-2:])
		elif li.count("Health:") > 0:
			card_dict["health"] = int(li[-2:])
		elif li.count("Durability:") > 0:
			card_dict["durability"] = int(li[-1])
		elif li.count("Minion Type") > 0:
			types = list()
			for minion_type in {"Amalgam", "Beast", "Demon", "Dragon", "Elemental", "Mech",
								"Murloc", "Naga", "Pirate", "Quilboar", "Totem", "Undead"}:
				if li.count(minion_type) > 0:
					types.append(minion_type)
			card_dict["minion_type"] = types
		elif li.count("School"):
			for spell_school in {"Arcane", "Fel", "Fire", "Holy", "Nature", "Shadow"}:
				if li.count(spell_school):
					card_dict["spell_school"] = spell_school
		elif li.count("Rarity"):
			for rarity in {"Common", "Rare", "Epic", "Legendary"}:
				if li.count(rarity) > 0:
					card_dict["rarity"] = rarity
		elif li.count("Class") > 0:
			classes = list()
			for hero_class in {"Druid", "Hunter", "Paladin", "Mage", "Warrior", "Shaman",
							   "Priest", "Demon Hunter", "Neutral", "Rogue", "Warlock",
							   "Death Knight"}:
				if li.count(hero_class) > 0:
					classes.append(hero_class)
			card_dict["class"] = classes
		elif li.count("Card Type") > 0:
			for card_type in {"Spell", "Weapon", "Hero", "Minion", "Location"}:
				if li.count(card_type) > 0:
					card_dict["card_type"] = card_type

	rating_text = str(soup.find("div", {"class": "gdrts-rating-text"}))
	rating_index_start = rating_text.find("<strong>")
	rating_index_end = rating_text.find("</strong>")
	mark = rating_text[rating_index_start + len("<strong>"):rating_index_end]
	card_dict["card_mark"] = float(mark)
	card_df = pd.DataFrame(columns=keys)
	card_df = card_df.append(card_dict, ignore_index=True)
	return card_df


def main():
	total_df = pd.DataFrame(columns=keys)

	for i in range(1, 86):
		print(f"Page number {i}/85")
		response = requests.get(base_url + str(i))
		soup = BeautifulSoup(response.text, "html.parser")
		cards = soup.findAll("div", class_="col-md-6 col-sm-12 col-xs-24")
		for card in cards:
			card_name = card.find("a").get("href").rsplit('/')[-2]
			card_url = "https://www.hearthstonetopdecks.com/cards/" + card_name + "/"
			card_df = parse_card_page(card_url, card_name)
			total_df = total_df._append(card_df)

	print(total_df)
	total_df.to_csv("HSTopdeck.csv", index=False)


if __name__ == "__main__":
	main()

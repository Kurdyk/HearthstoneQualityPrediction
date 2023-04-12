import requests
from bs4 import BeautifulSoup

base_url = "https://www.hearthstonetopdecks.com/cards/page/"

for i in range(1,86):
	response = requests.get(base_url + str(i))
	soup = BeautifulSoup(response.text,"html.parser")
	cards = soup.findAll("div", class_="col-md-6 col-sm-12 col-xs-24")
	for card in cards:
		card_name = card.find("a").get("href").rsplit('/')[-2]
		card_mark = card.find("div", class_="gdrts-rating-text").text
		print(card_name," : ",card_mark)

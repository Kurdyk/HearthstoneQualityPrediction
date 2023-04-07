def cleanDf(df):
	df = df[df.cardSet not in {"Missions","Credits","Hero Skins","Tavern Brawl","Mercenaries","Battlegrounds"}]
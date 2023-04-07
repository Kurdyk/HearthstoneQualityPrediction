import pandas as pd
import numpy as np


class DataVisualizer:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def filter_col(self, unwanted_col):
        self.df = self.df.drop(columns=unwanted_col)

    def filter_isin(self, varible, wanted_set):
        self.df = self.df[self.df[varible].isin(wanted_set)]

    def print(self):
        print(self.df)


if __name__ == "__main__":
    df = pd.read_csv("../hearthstone.csv")
    dv = DataVisualizer(df)
    print(df.keys())
    unwanted_col = ["cardId", "dbfId", "locale", "elite", "img", "flavor", "artist", "imgGold", "howToGetGold",
                    "howToGet", "howToGetDiamond", "faction", "classes"]
    dv.filter_col(unwanted_col)
    dv.filter_isin("Category", set(df["Category"].unique()) - {"Missions", "Credits", "Hero Skins", "Tavern Brawl",
                                                               "Battlegrounds", "Mercenaries"})
    dv.filter_isin("type", set(df["type"].unique()) - {"Hero Power", "Enchantment", np.nan, "Hero"})
    dv.filter_isin("collectible", set(df["collectible"].unique()) - {np.nan})
    dv.filter_col(["collectible"])
    dv.print()
    dv.df.to_csv("../filterData.csv")

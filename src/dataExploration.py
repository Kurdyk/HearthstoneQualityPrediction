import pandas as pd


class DataVisualizer:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def filter_col(self, unwanted_col):
        self.df = self.df.drop(columns=unwanted_col)

    def filter_isin(self, varible, wanted_set):
        self.df = self.df[self.df[varible].isin(wanted_set)]

    def print_col(self,columns):
        print(self.df[columns])

    def print(self):
        print(self.df)


if __name__ == "__main__":
    df = pd.read_csv("../OldData/hearthstone.csv")
    dv = DataVisualizer(df)
    unwanted_col = ["cardId", "dbfId", "locale", "elite", "img", "flavor", "artist", "imgGold", "howToGetGold",
                    "howToGet", "howToGetDiamond", "faction"]
    dv.filter_col(unwanted_col)

import pandas as pd


class DataVisualizer:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def filter_col(self, unwanted_col):
        self.df = self.df.drop(columns=unwanted_col)

    def filter_row(self, mask):
        self.df = self.df.loc[mask]


if __name__ == "__main__":
    df = pd.read_csv("../hearthstone.csv")
    dv = DataVisualizer(df)
    print(df.keys())
    unwanted_col = ["cardId", "dbfId", "locale", "elite", "img", "flavor", "artist", "imgGold", "howToGetGold",
                    "howToGet", "howToGetDiamond", "faction"]
    mask = df["Category"] in ["Basic", "Classic"]
    dv.filter_row(mask)
    # print(df["Category"].unique())

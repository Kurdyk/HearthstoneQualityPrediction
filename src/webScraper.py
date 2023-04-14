import pandas as pd


def reformat(path):
    in_file = open(path, "r")
    out_path = path + "_out"

    def put_newline():
        result = str()
        for line in in_file:
            for char in line:
                if char == ">":
                    result += ">\n"
                else:
                    result += char
        return result

    out_file = open(out_path, "w")
    out_file.write(put_newline())
    in_file.close()
    out_file.close()
    return out_path


def scrap_order(path):
    in_file = open(path, "r")
    in_list = list()
    for line in in_file:
        if line.count("</figcaption>") > 0:
            name = line.strip("</figcaption>\n")
            in_list.append(name)
    in_file.close()
    return in_list


def scrap_data(path):

    def parse_data(raw_data: str):
        if raw_data == "-":
            return 0
        if raw_data.count("%") > 0:
            return float(raw_data.strip("%")) / 100
        elif raw_data.count(",") > 0:
            striped = str()
            for c in raw_data:
                if c != ",":
                    striped += c
            return int(striped)
        else:
            return float(raw_data)

    in_file = open(path, "r")
    in_list = list()
    count = 0
    tmp = list()
    for line in in_file:
        if line.count("</div>") > 0:
            count += 1
            raw_data = line.strip("</div>\n")
            tmp.append(parse_data(raw_data))
        if count == 4:
            count = 0
            in_list.append(tmp.copy())
            tmp = list()
    return in_list


def fuse(name_list: list, data_list: list):
    fused_list = list()
    for name, data in zip(name_list, data_list):
        fused_list.append([name] + data)
    print(fused_list)
    result = pd.DataFrame(fused_list, columns=["name", "playRate", "avgCopies", "DeckWinrate", "TimesPlayed"])
    result.to_csv("../WebScrapping.csv")
    return result


if __name__ == "__main__":
    reformated_order = reformat("../OldData/WebCardsOrder")
    reformated_data = reformat("../OldData/WebScraping")
    fused_dataframe = fuse(scrap_order(reformated_order), scrap_data(reformated_data))
    first_dataframe = pd.read_csv("../OldData/filterData.csv")
    joined = first_dataframe.set_index("name").join(fused_dataframe.set_index("name"), on="name")
    print(joined)
    print(joined.keys())
    print(joined[joined["name"] == "Construct Quarter"])
    joined.to_csv("../full_data.csv")

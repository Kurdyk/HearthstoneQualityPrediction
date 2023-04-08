def reformat(path):

    in_file = open(path, "r")

    def put_newline():
        result = str()
        for line in in_file:
            for char in line:
                if char == ">":
                    result += ">\n"
                else:
                    result += char
        return result

    out_file = open(path + "_out", "w")
    out_file.write(put_newline())
    in_file.close()
    out_file.close()


if __name__ == "__main__":
    reformat("../WebCardsOrder")
    reformat("../WebScraping")

import pandas as pd

data = pd.read_excel("Excel Data.xlsx")

data = data[data["Position"].isin(["MF", "DF"])]

midfielders = data[data["Position"] == "MF"]["Completed Passes"]
defenders = data[data["Position"] == "DF"]["Completed Passes"]

mf_range = midfielders.max() - midfielders.min()
df_range = defenders.max() - defenders.min()

print("Midfielders")
print(midfielders.describe())

print("Defenders")
print(defenders.describe())


print("Midfielder Range:", mf_range)
print("Defender Range:", df_range)
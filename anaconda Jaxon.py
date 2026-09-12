import pandas as pd
from scipy import stats
import math

df = pd.read_excel("Excel Data.xlsx")

df = df[df["Position"].isin(["MF", "DF"])]

completed_passes = df["Completed Passes"]

n = len(completed_passes)
mean = completed_passes.mean()
std = completed_passes.std()

standard_error = std / math.sqrt(n)

confidence_interval = stats.t.interval(
    0.95,
    df=n - 1,
    loc=mean,
    scale=standard_error
)

print("Sample size:", n)
print("Mean:", mean)
print("Standard deviation:", std)
print("95% Confidence Interval:", confidence_interval)


midfielders = df[df["Position"] == "MF"]["Completed Passes"]
defenders = df[df["Position"] == "DF"]["Completed Passes"]

print("\nMidfielder mean:", midfielders.mean())
print("Defender mean:", defenders.mean())

t_stat, p_value = stats.ttest_ind(
    midfielders,
    defenders,
    equal_var=False
)

print("t-statistic:", t_stat)
print("p-value:", p_value)

if p_value < 0.05:
    print("Reject the null hypothesis.")
else:
    print("Fail to reject the null hypothesis.")
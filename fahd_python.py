import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


# Data Wrangling
# Import the goalkeeper dataset
df = pd.read_csv("goalkeepers.csv")

# Data Preparation and Sampling
# Include only goalkeepers who played at least two matches
df = df[df["Matches Played"] >= 2].copy()

# Calculate shots on target faced per match
df["SoTA per Match"] = df["SoTA"] / df["Matches Played"]

# Calculate the median SoTA per match
median_sota = df["SoTA per Match"].median()

print("Median SoTA per Match:", median_sota)

# Divide goalkeepers into High and Low SoTA groups
df["Group"] = df["SoTA per Match"].apply(
    lambda x: "High SoTA" if x >= median_sota else "Low SoTA"
)


# Descriptive Statistics
# Summarise Save% for each SoTA group
descriptive_stats = df.groupby("Group")["Save%"].agg(
    ["count", "mean", "median", "std", "min", "max"]
)

print("\nDescriptive Statistics:")
print(descriptive_stats)


# Define the Two Samples
# Response variable: Save%
# Explanatory variable: SoTA per Match Group
high_sota = df[df["Group"] == "High SoTA"]["Save%"].dropna()
low_sota = df[df["Group"] == "Low SoTA"]["Save%"].dropna()


# Inferential Statistics - Two-Sample t-Test
# H0: μHigh = μLow
# H1: μHigh ≠ μLow
# Significance level: α = 0.05

t_statistic, p_value = stats.ttest_ind(
    high_sota,
    low_sota,
    equal_var=False
)

print("\nWelch Two-Sample t-Test:")
print("t-statistic:", round(t_statistic, 4))
print("p-value:", round(p_value, 4))


# Inferential Statistics - 95% Confidence Interval
# Calculate the difference between the two mean Save% values
mean_high = high_sota.mean()
mean_low = low_sota.mean()
mean_difference = mean_high - mean_low

# Calculate the standard error
standard_error = np.sqrt(
    high_sota.var(ddof=1) / len(high_sota)
    + low_sota.var(ddof=1) / len(low_sota)
)

# Calculate degrees of freedom using the Welch-Satterthwaite method
degrees_freedom = (
    (
        high_sota.var(ddof=1) / len(high_sota)
        + low_sota.var(ddof=1) / len(low_sota)
    ) ** 2
) / (
    (high_sota.var(ddof=1) / len(high_sota)) ** 2
    / (len(high_sota) - 1)
    +
    (low_sota.var(ddof=1) / len(low_sota)) ** 2
    / (len(low_sota) - 1)
)

# Find the critical t-value for a 95% confidence interval
critical_t = stats.t.ppf(0.975, degrees_freedom)

# Calculate the lower and upper confidence interval bounds
lower_bound = mean_difference - critical_t * standard_error
upper_bound = mean_difference + critical_t * standard_error

print("\n95% Confidence Interval:")
print("Mean Save% - High SoTA:", round(mean_high, 2), "%")
print("Mean Save% - Low SoTA:", round(mean_low, 2), "%")
print("Mean Difference:", round(mean_difference, 2), "percentage points")
print("95% CI:", round(lower_bound, 2), "to", round(upper_bound, 2))


# Hypothesis Test Decision
alpha = 0.05

if p_value < alpha:
    print("\nDecision: Reject H0")
    print("There is a statistically significant difference in mean Save%.")
else:
    print("\nDecision: Fail to reject H0")
    print("There is no statistically significant difference in mean Save%.")


# Data Visualisation
# Create a boxplot comparing Save% between the two groups
plt.boxplot(
    [low_sota, high_sota],
    tick_labels=["Low SoTA", "High SoTA"]
)

plt.xlabel("Shots on Target Faced per Match Group")
plt.ylabel("Save Percentage (%)")
plt.title("Goalkeeper Save% by SoTA per Match Group")

plt.show()
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Data source: FIFA World Cup 2026 Official Matches page.
# Keep this Python file and data.csv in the same folder.
DATA_FILE = "data.csv"

# Read the CSV file.
df = pd.read_csv(DATA_FILE)

# Check for duplicate team-match records before starting the analysis.
duplicate_rows = df.duplicated(subset=["match_id", "team"]).sum()
print(f"Duplicate team-match records: {duplicate_rows}")

# Remove duplicate team-match records if there are any.
df = df.drop_duplicates(subset=["match_id", "team"])

# Only keep the two columns I need for this question.
analysis_data = df[["stage_group", "goals"]].dropna().copy()

# Take a stratified random sample.
# 50 records are selected from each stage, so the sample has 100 records.
analysis_data = (
    analysis_data.groupby("stage_group", group_keys=False)
    .sample(n=50, random_state=42)
)

print("\nSample size in each tournament stage:")
print(analysis_data["stage_group"].value_counts())

# Check if there are any missing values.
print("Missing values:")
print(analysis_data.isna().sum())

# Check how many records are in each stage.
print("\nNumber of team-match records:")
print(analysis_data["stage_group"].value_counts())

# Calculate basic statistics for group stage and knockout stage.
summary = (
    analysis_data.groupby("stage_group")["goals"]
    .agg(
        count="count",
        mean="mean",
        median="median",
        standard_deviation="std",
        minimum="min",
        maximum="max",
    )
    .round(2)
)

print("\nGoals scored by tournament stage:")
print(summary)

# Calculate a 95% confidence interval for the mean goals in each stage.
# It gives a lower and upper estimate for the true average goals.
confidence_interval_results = []

for stage in ["Group stage", "Knockout stage"]:
    stage_goals = analysis_data.loc[
        analysis_data["stage_group"] == stage, "goals"
    ]

    sample_size = len(stage_goals)
    mean_goals = stage_goals.mean()
    standard_error = stats.sem(stage_goals)
    t_critical_value = stats.t.ppf(0.975, df=sample_size - 1)
    margin_of_error = t_critical_value * standard_error

    lower_limit = mean_goals - margin_of_error
    upper_limit = mean_goals + margin_of_error

    confidence_interval_results.append(
        {
            "stage_group": stage,
            "mean_goals": mean_goals,
            "lower_95_CI": lower_limit,
            "upper_95_CI": upper_limit,
        }
    )

    print(f"\n{stage} 95% confidence interval:")
    print(f"({lower_limit:.2f}, {upper_limit:.2f})")

# Save the confidence-interval results for the report or presentation.
confidence_intervals = pd.DataFrame(confidence_interval_results).round(2)
confidence_intervals.to_csv(
    "confidence_intervals.csv",
    index=False
)
print("\nSaved: confidence_intervals.csv")

# Save the results so I can use them in my presentation.
summary.to_csv("descriptive_statistics_by_stage.csv")
print("\nSaved: descriptive_statistics_by_stage.csv")

# Make a boxplot to show the spread of goals in the two stages.
group_goals = analysis_data.loc[
    analysis_data["stage_group"] == "Group stage", "goals"
]
knockout_goals = analysis_data.loc[
    analysis_data["stage_group"] == "Knockout stage", "goals"
]

# Use a two-sample t-test to compare the average goals in the two stages.
# The null hypothesis is that the two average numbers of goals are the same.
t_statistic, p_value = stats.ttest_ind(
    group_goals,
    knockout_goals,
    equal_var=False
)

alpha = 0.05

print("\nTwo-sample t-test results:")
print(f"t-statistic: {t_statistic:.3f}")
print(f"p-value: {p_value:.4f}")

if p_value < alpha:
    conclusion = "Reject the null hypothesis: the average goals are significantly different."
else:
    conclusion = "Fail to reject the null hypothesis: there is no significant difference in average goals."

print(conclusion)

# Save the t-test result for the report or presentation.
t_test_results = pd.DataFrame(
    {
        "t_statistic": [round(t_statistic, 3)],
        "p_value": [round(p_value, 4)],
        "significance_level": [alpha],
        "conclusion": [conclusion],
    }
)
t_test_results.to_csv(
    "t_test_results.csv",
    index=False
)
print("Saved: t_test_results.csv")

plt.figure(figsize=(8, 5))
plt.boxplot(
    [group_goals, knockout_goals],
    tick_labels=["Group stage", "Knockout stage"]
)
plt.title("Goals scored in group stage and knockout stage")
plt.xlabel("Tournament stage")
plt.ylabel("Goals scored by a team in one match")
plt.tight_layout()
plt.savefig(
    "boxplot_goals_by_stage.png",
    dpi=300
)
plt.show()

# Make a bar chart to compare the average goals in the two stages.
average_goals = analysis_data.groupby("stage_group")["goals"].mean()

plt.figure(figsize=(8, 5))
plt.bar(
    average_goals.index,
    average_goals.values,
    color=["skyblue", "orange"]
)
plt.title("Average goals scored by tournament stage")
plt.xlabel("Tournament stage")
plt.ylabel("Average goals scored by a team in one match")
plt.ylim(0, max(average_goals.values) + 0.5)
plt.tight_layout()
plt.savefig(
    "bar_chart_average_goals.png",
    dpi=300
)
plt.show()

print("Saved: boxplot_goals_by_stage.png")
print("Saved: bar_chart_average_goals.png")

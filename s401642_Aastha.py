import os
import sys
import pandas as pd
import numpy as np
from scipy import stats

# Ensures Windows terminal output handle characters without errors
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# -------------------------------------------------------------------------
# 1. AUTOMATION & DATA WRANGLING 
# -------------------------------------------------------------------------
def automated_football_pipeline(file_path):
    print(f"[INFO] Ingesting dataset from source path: {file_path}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"[ERROR] Target CSV file '{file_path}' could not be located.")
        
    df = pd.read_csv(file_path)
    
    
    df.columns = df.columns.str.strip().str.lower()
    print(f"[DEBUG 1] Columns found in your CSV file: {df.columns.tolist()}")
    
    # Isolate only the exact critical columns needed for this specific T-Test calculation
    # This prevents the code from dropping rows due to missing cells in unrelated columns
    critical_columns = ['shots on target', 'total shots', 'match stage']
    df = df.dropna(subset=critical_columns)
    
    # Standardisation: Clean and standardise text casing for easy sorting
    df['match stage'] = df['match stage'].astype(str).str.strip().str.title()
    
    # Type Safety Implementation via Explicit Casting
    df['shots on target'] = pd.to_numeric(df['shots on target'], errors='coerce').fillna(0).astype(int)
    df['total shots'] = pd.to_numeric(df['total shots'], errors='coerce').fillna(1).astype(int)
    
    
    # Minor epsilon variance adjustment (1e-5) safely isolates against ZeroDivisionErrors
    df['accuracy_ratio'] = df['shots on target'] / (df['total shots'] + 1e-5)
    
    print("[SUCCESS] Ingestion, standardisation, and feature engineering complete.")
    return df


csv_filename = "Final_shots_data_Aastha.csv"
df_processed = automated_football_pipeline(csv_filename)
print(repr(df_processed['match stage'].unique()))

# -------------------------------------------------------------------------
# 2. DATA PREPARATION & SAMPLING 
# -------------------------------------------------------------------------

population_group = df_processed[df_processed['match stage'].str.contains('Group|Grp|G', na=False, case=False)]
population_knockout = df_processed[
    df_processed['match stage'].str.contains('Knockout', na=False, case=False)
]

print(f"[DEBUG 2] Group matches available: {len(population_group)}")
print(f"[DEBUG 3] Knockout matches available: {len(population_knockout)}")

# Set random seed to ensure strict statistical reproducibility
np.random.seed(42)

# Selects a clean sample size dynamically up to n=30 to satisfy Central Limit Theorem (CLT)
sample_size = min(len(population_group), len(population_knockout), 30)

# Failsafe abort check to provide explicit feedback if your knockout data isn't logged yet
if sample_size == 0:
    print("\n[IMPORTANT NOTICE] Your Group stage rows loaded perfectly!")
    print(f"Group Stage Count: {len(population_group)} rows matches.")
    print(f"Knockout Stage Count: {len(population_knockout)} rows matches.")
    print("Note: If Knockout is 0, make sure to add your Knockout match rows at the bottom of your Excel sheet!")
    sys.exit()

print(f"[INFO] Successfully drawing a random sample of n={sample_size} matches per stage.")

sample_group = population_group.sample(n=sample_size, replace=False)
sample_knockout = population_knockout.sample(n=sample_size, replace=False)

# Build internal metrics arrays
group_metrics = sample_group['accuracy_ratio']
knockout_metrics = sample_knockout['accuracy_ratio']

# -------------------------------------------------------------------------
# 3. DESCRIPTIVE STATISTICS
# -------------------------------------------------------------------------
print("\n" + "="*50)
print("=== SECTION 3: DESCRIPTIVE STATISTICS OVERVIEW ===")
print("="*50)

print(f"Tournament Population Baseline Accuracy Mean:     {df_processed['accuracy_ratio'].mean():.4f}")
print("-" * 50)

print(f"Group Stage Random Sample (n={sample_size}) Accuracy Mean: {group_metrics.mean():.4f}")
print(f"Knockout Stage Random Sample (n={sample_size}) Accuracy Mean: {knockout_metrics.mean():.4f}")


# -------------------------------------------------------------------------
# 4. INFERENTIAL STATISTICS: ESTIMATED POPULATION CONFIDENCE INTERVALS
# -------------------------------------------------------------------------
print("\n" + "="*50)
print("=== SECTION 4: INFERENTIAL CONFIDENCE INTERVALS (95%) ===")
print("="*50)

# Degrees of Freedom (df = n - 1)
df_group = len(group_metrics) - 1
df_knockout = len(knockout_metrics) - 1

#  Standard Error of the Mean (SEM)
sem_group = stats.sem(group_metrics)
sem_knockout = stats.sem(knockout_metrics)

# 95% Confidence Intervals
ci_group = stats.t.interval(0.95, df=df_group, loc=group_metrics.mean(), scale=sem_group)
ci_knockout = stats.t.interval(0.95, df=df_knockout, loc=knockout_metrics.mean(), scale=sem_knockout)

# Print all the statistical calculated values
print(f"Group Stage Degrees of Freedom (df):               {df_group}")
print(f"Group Stage Standard Error of Mean (SEM):          {sem_group:.4f}")

print(f"Group Stage Mean Accuracy 95% Confidence Interval:  ({ci_group[0]:.4f}, {ci_group[1]:.4f})")
print("-" * 50)
print(f"Knockout Stage Degrees of Freedom (df):            {df_knockout}")
print(f"Knockout Stage Standard Error of Mean (SEM):       {sem_knockout:.4f}")

print(f"Knockout Stage Mean Accuracy 95% Confidence Interval: ({ci_knockout[0]:.4f}, {ci_knockout[1]:.4f})")


# -------------------------------------------------------------------------
# 5. INFERENTIAL STATISTICS: TWO-SAMPLE HYPOTHESIS TESTING (WELCH'S T-TEST)
# -------------------------------------------------------------------------
print("\n" + "="*50)
print("=== SECTION 5: TWO-SAMPLE PARAMETRIC T-TEST ===")
print("="*50)

t_stat, p_value = stats.ttest_ind(knockout_metrics, group_metrics, equal_var=False)

print(f"Calculated Welch's T-Statistic value: {t_stat:.4f}")
print(f"Calculated Empirical P-Value:         {p_value:.6f}")

alpha = 0.05
print("\n[ANALYSIS] STATISTICAL INTERPRETATION & ANALYSIS:")
if p_value < alpha:
    print(f"Result: Reject H0 as P-value ({p_value:.6f}) < Alpha ({alpha}).")
    print("Conclusion: There is a statistically significant variance in shooting accuracy ratios between tournament stages.")
else:
    print(f"Result: Fail to Reject H0 as P-value ({p_value:.6f}) >= Alpha ({alpha}).")
    print("Conclusion: There is insufficient empirical evidence to prove that tournament stage environments alter shooting accuracy parameters.")



import matplotlib.pyplot as plt
import seaborn as sns


sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 14})

# Create a combined helper dataframe from your random samples for easy plotting
sample_group_df = sample_group.copy()
sample_group_df['Stage_Label'] = 'Group'

sample_knockout_df = sample_knockout.copy()
sample_knockout_df['Stage_Label'] = 'Knockout'

plot_df = pd.concat([sample_group_df, sample_knockout_df], axis=0)

# =========================================================================
# GRAPH 1: GROUPED BOX PLOT (Distribution & Outlier Tracking)
# =========================================================================
plt.figure(figsize=(8, 6))
sns.boxplot(
    data=plot_df, 
    x='Stage_Label', 
    y='accuracy_ratio', 
    palette='Set2', 
    width=0.5
)
# Overlay individual match points transparently to see exact data density
sns.stripplot(
    data=plot_df, 
    x='Stage_Label', 
    y='accuracy_ratio', 
    color='black', 
    alpha=0.3, 
    jitter=0.1
)

plt.title('Shooting Accuracy Ratio Distribution: Group vs. Knockout Stage')
plt.xlabel('Tournament Match Stage')
plt.ylabel('Shooting Accuracy Ratio (Shots on Target / Total Shots)')
plt.ylim(0, 1.0) # Accuracy ratio constraints range logically from 0% to 100%
plt.tight_layout()
plt.show()
# =========================================================================
# GRAPH 2: OVERLAPPING KERNEL DENSITY ESTIMATE (Normality Verification)
# =========================================================================
plt.figure(figsize=(9, 5))
sns.kdeplot(
    data=plot_df, 
    x='accuracy_ratio', 
    hue='Stage_Label', 
    fill=True, 
    common_norm=False, 
    palette='Set1', 
    alpha=0.4, 
    linewidth=2
)

plt.title('Sample Density Distribution of Shooting Accuracy Ratios')
plt.xlabel('Shooting Accuracy Ratio')
plt.ylabel('Probability Density Frequency')
plt.xlim(0, 1.0)
plt.tight_layout()
plt.show()

# =========================================================================
# GRAPH 3: SAMPLE MEANS WITH 95% CONFIDENCE INTERVAL ERROR BARS
# =========================================================================
plt.figure(figsize=(7, 6))

# Extract means and calculate standard error lengths for the error bars
means = [group_metrics.mean(), knockout_metrics.mean()]
stages = ['Group', 'Knockout']

# Calculate explicit error bar lengths using standard error of the mean (SEM)
# Multiplied by 1.96 to accurately visually match a standard 95% normal distribution boundary
error_lengths = [stats.sem(group_metrics) * 1.96, stats.sem(knockout_metrics) * 1.96]

plt.errorbar(
    x=stages, 
    y=means, 
    yerr=error_lengths, 
    fmt='o',         # Plot means as solid round points
    color='darkblue', 
    markersize=8, 
    capsize=8,       # Horizontal caps on top/bottom of error bar brackets
    linewidth=2, 
    label='Sample Mean with 95% CI'
)

plt.title('Comparison of Mean Shooting Accuracy with 95% Confidence Intervals')
plt.xlabel('Tournament Match Stage')
plt.ylabel('Mean Shooting Accuracy Ratio')
plt.ylim(0, 1.0)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the SkillMatch dataset
df = pd.read_csv(
    'C:/Users/AMAN/Documents/Skill_Match/dataset/reordered_raw.csv'
)

# 2. Retrieve data in different ways

# View the first 5 rows
print("--- First 5 Rows ---")
print(df.head())

# Print specific columns
print("---- Print First 6 Columns ----")
subset = df.iloc[:, 0:6]
print(subset)

# 3. Identify missing values per column
missing_counts = df.isnull().sum()

print("----- Missing Values Per Column -----")
print(missing_counts)

# 4. Total missing values
total_missing = df.isnull().sum().sum()

print("-----------------------------------")
print("Total Missing Values:", total_missing)
print("-" * 40)

# 5. Detect duplicate rows
duplicate_rows = df[df.duplicated()]

print(f"Total duplicate rows detected: {len(duplicate_rows)}")
print(duplicate_rows)
print("-" * 40)

# 6. Produce a missingness heatmap
plt.figure(figsize=(10, 6))

sns.heatmap(
    df.isnull(),
    cbar=False,
    yticklabels=False,
    cmap="viridis"
)

plt.title("SkillMatch - Missing Values Heatmap")
plt.show()
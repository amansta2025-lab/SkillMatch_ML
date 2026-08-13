import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
import matplotlib.pyplot as plt


# ============================================================
# Step 1: Load Dataset
# ============================================================

file_path = "C:/Users/AMAN/Documents/Skill_Match/dataset/reordered_raw.csv"

df = pd.read_csv(file_path)

print("Original Dataset")
print("------------------------")
print(df.head())

print("\nDataset Shape:", df.shape)

print("\nData Types:")
print("------------------------")
print(df.dtypes)

print("\nMissing Values:")
print("------------------------")
print(df.isnull().sum())

print("\nDuplicate Records:", df.duplicated().sum())


# ============================================================
# Step 2: Remove Duplicate Records
# ============================================================

df = df.drop_duplicates()

print("\nDataset Shape After Removing Duplicates:", df.shape)


# ============================================================
# Step 3: Handle Missing Values
# ============================================================

# Numerical Columns
numerical_columns = df.select_dtypes(
    include=['int64', 'float64']
).columns

# Fill missing numerical values with mean
for column in numerical_columns:
    df[column] = df[column].fillna(df[column].mean())


# Categorical Columns
categorical_columns = df.select_dtypes(
    include=['object', 'str']
).columns

# Fill missing categorical values with mode
for column in categorical_columns:
    df[column] = df[column].fillna(df[column].mode()[0])


# ============================================================
# Step 4: Remove Leading and Trailing Spaces
# ============================================================

for column in categorical_columns:
    df[column] = df[column].str.strip()


# ============================================================
# Step 5: Remove Useless Columns
# ============================================================

# These columns do not provide useful information
# for machine learning.

columns_to_remove = [
    'EmployeeCount',
    'EmployeeNumber',
    'StandardHours',
    'Over18'
]

# Remove only columns that actually exist
columns_to_remove = [
    column for column in columns_to_remove
    if column in df.columns
]

df = df.drop(columns=columns_to_remove)

print("\nRemoved Useless Columns:")
print(columns_to_remove)


# ============================================================
# Step 6: Select Numeric Columns
# ============================================================

numeric_columns = df.select_dtypes(
    include=['int64', 'float64']
).columns

print("\nNumeric Columns:")
print(list(numeric_columns))


# ============================================================
# Step 7: Standardization (Z-score)
# Mean = 0
# Standard Deviation = 1
# ============================================================

standard_scaler = StandardScaler()

standardized = standard_scaler.fit_transform(
    df[numeric_columns]
)

standardized_df = pd.DataFrame(
    standardized,
    columns=[
        col + "_Standardized"
        for col in numeric_columns
    ],
    index=df.index
)


# ============================================================
# Step 8: Min-Max Scaling
# Values between 0 and 1
# ============================================================

minmax_scaler = MinMaxScaler()

scaled = minmax_scaler.fit_transform(
    df[numeric_columns]
)

scaled_df = pd.DataFrame(
    scaled,
    columns=[
        col + "_Scaled"
        for col in numeric_columns
    ],
    index=df.index
)


# ============================================================
# Step 9: Normalization (L2 Normalization)
# Each row becomes a unit vector
# ============================================================

normalizer = Normalizer(norm='l2')

normalized = normalizer.fit_transform(
    df[numeric_columns]
)

normalized_df = pd.DataFrame(
    normalized,
    columns=[
        col + "_Normalized"
        for col in numeric_columns
    ],
    index=df.index
)


# ============================================================
# Step 10: Combine Original + Processed Columns
# ============================================================

df = pd.concat(
    [
        df,
        standardized_df,
        scaled_df,
        normalized_df
    ],
    axis=1
)


# ============================================================
# Step 11: Display Results After Preprocessing
# ============================================================

print("\nDisplay Results After Preprocessed Dataset")
print("--------------------------------------------")

print(df.head())

print("\nDataset Shape:", df.shape)

print("\nDataset Information:")
print("------------------------")
df.info()

print("\nColumns in Dataset:")
print(df.columns.tolist())

print("\nMissing Values After Preprocessing:")
print(df.isnull().sum())

print("\nDuplicate Records After Preprocessing:")
print(df.duplicated().sum())


# ============================================================
# Step 12: Save Preprocessed Dataset
# ============================================================

output_path = "C:/Users/AMAN/Documents/Skill_Match/dataset/clean_minmax_stand_norma_M2.csv"

df.to_csv(
    output_path,
    index=False
)

print("\nPreprocessed dataset saved successfully.")
print("Saved to:", output_path)


# ============================================================
# Step 13: Read Preprocessed Dataset
# ============================================================

pf = pd.read_csv(output_path)


# ============================================================
# Step 14: Display Histograms
# ============================================================

pf.hist(
    figsize=(15, 12),
    bins=10,
    edgecolor='black'
)

plt.suptitle(
    "Histogram of Preprocessed Employee Attrition Dataset"
)

plt.tight_layout()

plt.show()
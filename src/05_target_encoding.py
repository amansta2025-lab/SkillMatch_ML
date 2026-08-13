import pandas as pd
from category_encoders import TargetEncoder
from pathlib import Path


# ============================================================
# 1. FILE PATHS
# ============================================================



INPUT_FILE ="C:/Users/AMAN/Documents/Skill_Match/dataset/reordered_raw.csv"
OUTPUT_FILE = "C:/Users/AMAN/Documents/Skill_Match/dataset/target_encoding.csv"


# ============================================================
# 2. TARGET COLUMN
# ============================================================

TARGET_COLUMN = "Attrition"


# ============================================================
# 3. READ DATASET
# ============================================================

df = pd.read_csv(INPUT_FILE)

print("=" * 60)
print("DATASET LOADED")
print("=" * 60)

print("Rows    :", df.shape[0])
print("Columns :", df.shape[1])

print("\nTarget column:", TARGET_COLUMN)


# ============================================================
# 4. CHECK TARGET COLUMN
# ============================================================

if TARGET_COLUMN not in df.columns:
    raise ValueError(
        f"Target column '{TARGET_COLUMN}' not found in dataset."
    )


# ============================================================
# 5. CONVERT TARGET TO NUMERIC
# ============================================================

df[TARGET_COLUMN] = (
    df[TARGET_COLUMN]
    .astype(str)
    .str.strip()
    .map({
        "No": 0,
        "Yes": 1
    })
)


print("\nTarget values:")
print(df[TARGET_COLUMN].value_counts(dropna=False))


# ============================================================
# 6. FIND CATEGORICAL COLUMNS
# ============================================================

categorical_columns = []

for column in df.columns:

    if column == TARGET_COLUMN:
        continue

    if df[column].dtype == "object" or \
       pd.api.types.is_string_dtype(df[column]):

        categorical_columns.append(column)


print("\n" + "=" * 60)
print("CATEGORICAL COLUMNS")
print("=" * 60)

for column in categorical_columns:
    print(column)


# ============================================================
# 7. TARGET ENCODING
# ============================================================

encoder = TargetEncoder(
    cols=categorical_columns,
    handle_missing="value",
    handle_unknown="value"
)


df[categorical_columns] = encoder.fit_transform(
    df[categorical_columns],
    df[TARGET_COLUMN]
)


# ============================================================
# 8. SAVE ENCODED DATASET
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 9. FINAL INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("TARGET ENCODING COMPLETED")
print("=" * 60)

print("Original rows    :", 1470)
print("Original columns :", 35)

print("Output rows      :", df.shape[0])
print("Output columns   :", df.shape[1])

print("\nOutput file:")
print(OUTPUT_FILE)

print("\nFirst 5 rows:")
print(df.head())

print("\nFile saved successfully!")
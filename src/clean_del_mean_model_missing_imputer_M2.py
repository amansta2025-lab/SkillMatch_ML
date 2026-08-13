# ============================================================
# CLEAN DEL MEAN MODEL MISSING IMPUTER - M2
# Skill Match Dataset
# ============================================================

import pandas as pd
from pathlib import Path
from sklearn.impute import SimpleImputer


# ============================================================
# 1. PROJECT PATH
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_FOLDER = PROJECT_DIR / "dataset"

OUTPUT_FILE = DATASET_FOLDER / "cleaned_mean_model_imputed.csv"


# ============================================================
# 2. FIND ORIGINAL CSV
# ============================================================

csv_files = [
    file for file in DATASET_FOLDER.glob("*.csv")
    if file.name != "cleaned_mean_model_imputed.csv"
]


if len(csv_files) == 0:
    raise FileNotFoundError(
        f"No CSV file found inside:\n{DATASET_FOLDER}\n\n"
        "Please place your original dataset inside the dataset folder."
    )


print("=" * 60)
print("CSV FILES FOUND")
print("=" * 60)

for i, file in enumerate(csv_files, start=1):
    print(i, "->", file.name)


# ============================================================
# 3. SELECT ORIGINAL DATASET
# ============================================================

INPUT_FILE = csv_files[0]

print("\nUsing dataset:")
print(INPUT_FILE)


# ============================================================
# 4. READ DATASET
# ============================================================

df = pd.read_csv(INPUT_FILE)

print("\n" + "=" * 60)
print("ORIGINAL DATASET")
print("=" * 60)

print("Rows    :", df.shape[0])
print("Columns :", df.shape[1])


# ============================================================
# 5. MISSING VALUES BEFORE IMPUTATION
# ============================================================

print("\nMissing values BEFORE imputation:")

missing_before = df.isnull().sum()

missing_before = missing_before[
    missing_before > 0
]

if len(missing_before) == 0:
    print("No missing values found.")
else:
    print(missing_before)


# ============================================================
# 6. IDENTIFY NUMERIC COLUMNS
# ============================================================

numeric_columns = df.select_dtypes(
    include="number"
).columns.tolist()


# ============================================================
# 7. IDENTIFY CATEGORICAL COLUMNS
# ============================================================

categorical_columns = [
    column
    for column in df.columns
    if column not in numeric_columns
]


print("\nNumeric columns:")
print(numeric_columns)

print("\nCategorical columns:")
print(categorical_columns)


# ============================================================
# 8. NUMERIC IMPUTATION
#    Missing numeric values -> MEAN
# ============================================================

if numeric_columns:

    numeric_imputer = SimpleImputer(
        strategy="mean"
    )

    df[numeric_columns] = numeric_imputer.fit_transform(
        df[numeric_columns]
    )


# ============================================================
# 9. CATEGORICAL IMPUTATION
#    Missing categorical values -> MOST FREQUENT
# ============================================================

if categorical_columns:

    categorical_imputer = SimpleImputer(
        strategy="most_frequent"
    )

    df[categorical_columns] = categorical_imputer.fit_transform(
        df[categorical_columns]
    )


# ============================================================
# 10. CHECK MISSING VALUES AFTER IMPUTATION
# ============================================================

print("\nMissing values AFTER imputation:")

missing_after = df.isnull().sum()

missing_after = missing_after[
    missing_after > 0
]

if len(missing_after) == 0:
    print("No missing values remaining.")
else:
    print(missing_after)


# ============================================================
# 11. SAVE CLEANED DATASET
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 12. FINAL RESULT
# ============================================================

print("\n" + "=" * 60)
print("MISSING VALUE IMPUTATION COMPLETED")
print("=" * 60)

print("Input shape :", df.shape)
print("Output shape:", df.shape)

print("\nSaved file:")
print(OUTPUT_FILE)

print("\nFirst 5 rows:")
print(df.head())

print("\nFile saved successfully!")
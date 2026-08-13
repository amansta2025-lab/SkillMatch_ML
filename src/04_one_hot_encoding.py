import pandas as pd


# ============================================================
# FILE PATH
# ============================================================

INPUT_FILE ="C:/Users/AMAN/Documents/Skill_Match/dataset/reordered_raw.csv"

OUTPUT_FILE = "C:/Users/AMAN/Documents/Skill_Match/dataset/one_hot_encoding.csv"


# ============================================================
# READ DATASET
# ============================================================

df = pd.read_csv(INPUT_FILE)

print("=" * 60)
print("DATASET LOADED")
print("=" * 60)

print("Rows    :", df.shape[0])
print("Columns :", df.shape[1])


# ============================================================
# CATEGORICAL COLUMNS
# ============================================================

categorical_columns = [
    "BusinessTravel",
    "Department",
    "EducationField",
    "Gender",
    "JobRole",
    "MaritalStatus",
    "Over18",
    "OverTime",
    "Attrition"
]


# ============================================================
# ONE-HOT ENCODING
# ============================================================

df = pd.get_dummies(
    df,
    columns=categorical_columns,
    dtype=int
)


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# RESULT
# ============================================================

print("\nOne-Hot Encoding completed!")

print("Output shape:", df.shape)

print("\nOutput file:")
print(OUTPUT_FILE)

print("\nFirst 5 rows:")
print(df.head())
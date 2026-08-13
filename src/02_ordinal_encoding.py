import pandas as pd
from sklearn.preprocessing import OrdinalEncoder


# ============================================================
# FILE PATH
# ============================================================

INPUT_FILE ="C:/Users/AMAN/Documents/Skill_Match/dataset/reordered_raw.csv"

OUTPUT_FILE = "C:/Users/AMAN/Documents/Skill_Match/dataset/ordinal_encoding.csv"


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
# ORDINAL ENCODING
# ============================================================

encoder = OrdinalEncoder(
    handle_unknown="use_encoded_value",
    unknown_value=-1
)

df[categorical_columns] = encoder.fit_transform(
    df[categorical_columns].astype(str)
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

print("\nOrdinal Encoding completed!")

print("Output shape:", df.shape)

print("\nOutput file:")
print(OUTPUT_FILE)

print("\nFirst 5 rows:")
print(df.head())
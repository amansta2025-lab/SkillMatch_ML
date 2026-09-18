# ================================================================
# LightGBM Classifier - Employee Attrition Prediction
# ================================================================


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import joblib

from lightgbm import LGBMClassifier

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ================================================================
# 1. DATASET PATH
# ================================================================

DATASET_PATH = (
    "C:/Users/AMAN/Documents/Skill_Match/"
    "dataset/reordered_raw.csv"
)


# ================================================================
# 2. OUTPUT FOLDER
# ================================================================

OUTPUT_FOLDER = (
    "C:/Users/AMAN/Documents/Skill_Match/"
    "output/LightGBM_Classifier_M3_Outputs"
)


os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)


# ================================================================
# 3. LOAD DATASET
# ================================================================

df = pd.read_csv(
    DATASET_PATH
)


print(
    "Original Dataset Shape:",
    df.shape
)


# ================================================================
# 4. CREATE COPY
# ================================================================

processed_df = df.copy()


# ================================================================
# 5. REMOVE DUPLICATES
# ================================================================

processed_df.drop_duplicates(
    inplace=True
)


print(
    "Dataset Shape After Removing Duplicates:",
    processed_df.shape
)


# ================================================================
# 6. REMOVE UNNECESSARY COLUMNS
# ================================================================

# Remove index column if present
if "index" in processed_df.columns:

    processed_df.drop(
        columns=["index"],
        inplace=True
    )


# EmployeeNumber is only an identification number
if "EmployeeNumber" in processed_df.columns:

    processed_df.drop(
        columns=["EmployeeNumber"],
        inplace=True
    )


# ================================================================
# 7. FIND TARGET COLUMN
# ================================================================

target_candidates = [
    "Attrition",
    "attrition",
    "EmployeeAttrition"
]


target_column = None


for column in target_candidates:

    if column in processed_df.columns:

        target_column = column

        break


if target_column is None:

    raise ValueError(
        "Attrition target column not found."
    )


print(
    "\nTarget Column:",
    target_column
)


# ================================================================
# 8. HANDLE MISSING VALUES
# ================================================================

numeric_cols = processed_df.select_dtypes(
    include=["int64", "float64"]
).columns


for col in numeric_cols:

    processed_df[col] = processed_df[col].fillna(
        processed_df[col].median()
    )


categorical_cols = processed_df.select_dtypes(
    include=["object"]
).columns


for col in categorical_cols:

    processed_df[col] = processed_df[col].fillna(
        processed_df[col].mode()[0]
    )


# ================================================================
# 9. CLEAN CATEGORICAL DATA
# ================================================================

for col in categorical_cols:

    processed_df[col] = (
        processed_df[col]
        .astype(str)
        .str.strip()
        .str.lower()
    )


# ================================================================
# 10. SEPARATE FEATURES AND TARGET
# ================================================================

X = processed_df.drop(
    columns=[target_column]
)


y = processed_df[target_column]


# ================================================================
# 11. ENCODE CATEGORICAL FEATURES
# ================================================================

encoders = {}


for col in X.select_dtypes(
    include=["object"]
).columns:

    encoder = LabelEncoder()

    X[col] = encoder.fit_transform(
        X[col].astype(str)
    )

    encoders[col] = encoder


# ================================================================
# 12. ENCODE TARGET
# ================================================================

target_encoder = LabelEncoder()


y = target_encoder.fit_transform(
    y.astype(str)
)


print(
    "\nTarget Classes:"
)


print(
    target_encoder.classes_
)


# ================================================================
# 13. TRAIN-TEST SPLIT
# ================================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


print(
    "\nTraining Data:",
    X_train.shape
)


print(
    "Testing Data :",
    X_test.shape
)


# ================================================================
# 14. CALCULATE CLASS WEIGHT
# ================================================================

negative_count = (
    y_train == 0
).sum()


positive_count = (
    y_train == 1
).sum()


if positive_count > 0:

    scale_pos_weight = (
        negative_count /
        positive_count
    )

else:

    scale_pos_weight = 1


print(
    "\nNegative Samples:",
    negative_count
)


print(
    "Positive Samples:",
    positive_count
)


print(
    "Scale Pos Weight:",
    scale_pos_weight
)


# ================================================================
# 15. CREATE LIGHTGBM MODEL
# ================================================================

model = LGBMClassifier(

    n_estimators=200,

    learning_rate=0.05,

    max_depth=5,

    num_leaves=31,

    min_child_samples=20,

    subsample=0.8,

    colsample_bytree=0.8,

    reg_alpha=0,

    reg_lambda=1,

    scale_pos_weight=scale_pos_weight,

    objective="binary",

    random_state=42,

    n_jobs=-1,

    verbosity=-1
)


# ================================================================
# 16. TRAIN MODEL
# ================================================================

print(
    "\nTraining LightGBM Model..."
)


model.fit(
    X_train,
    y_train
)


print(
    "Model Training Completed!"
)


# ================================================================
# 17. MAKE PREDICTIONS
# ================================================================

y_pred = model.predict(
    X_test
)


# ================================================================
# 18. CALCULATE PERFORMANCE
# ================================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)


precision = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)


recall = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)


f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)


print(
    "\n================================================"
)


print(
    "          LIGHTGBM MODEL PERFORMANCE"
)


print(
    "================================================"
)


print(
    "Accuracy  :",
    accuracy
)


print(
    "Precision :",
    precision
)


print(
    "Recall    :",
    recall
)


print(
    "F1 Score  :",
    f1
)


# ================================================================
# 19. SAVE PERFORMANCE METRICS
# ================================================================

metrics_df = pd.DataFrame({

    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ],

    "Score": [
        accuracy,
        precision,
        recall,
        f1
    ]

})


metrics_df.to_csv(

    os.path.join(
        OUTPUT_FOLDER,
        "LightGBM_Performance_Metrics.csv"
    ),

    index=False
)


# ================================================================
# 20. CLASSIFICATION REPORT
# ================================================================

report = classification_report(

    y_test,

    y_pred,

    target_names=[
        str(x)
        for x in target_encoder.classes_
    ],

    zero_division=0
)


print(
    "\nClassification Report:"
)


print(
    report
)


with open(

    os.path.join(
        OUTPUT_FOLDER,
        "Classification_Report.txt"
    ),

    "w"
) as file:

    file.write(
        report
    )


# ================================================================
# 21. CONFUSION MATRIX
# ================================================================

cm = confusion_matrix(
    y_test,
    y_pred
)


print(
    "\nConfusion Matrix:"
)


print(
    cm
)


plt.figure(
    figsize=(6, 5)
)


plt.imshow(
    cm,
    interpolation="nearest"
)


plt.title(
    "LightGBM Confusion Matrix"
)


plt.colorbar()


plt.xticks(
    range(len(target_encoder.classes_)),
    target_encoder.classes_
)


plt.yticks(
    range(len(target_encoder.classes_)),
    target_encoder.classes_
)


plt.xlabel(
    "Predicted Label"
)


plt.ylabel(
    "Actual Label"
)


for i in range(cm.shape[0]):

    for j in range(cm.shape[1]):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )


plt.tight_layout()


plt.savefig(

    os.path.join(
        OUTPUT_FOLDER,
        "LightGBM_Confusion_Matrix.png"
    )

)


plt.close()


# ================================================================
# 22. PERFORMANCE GRAPH
# ================================================================

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score"
]


scores = [
    accuracy,
    precision,
    recall,
    f1
]


plt.figure(
    figsize=(8, 5)
)


plt.bar(
    metrics,
    scores
)


plt.ylim(
    0,
    1
)


plt.title(
    "LightGBM Model Performance"
)


plt.ylabel(
    "Score"
)


for i, value in enumerate(scores):

    plt.text(
        i,
        value + 0.02,
        f"{value:.3f}",
        ha="center"
    )


plt.tight_layout()


plt.savefig(

    os.path.join(
        OUTPUT_FOLDER,
        "LightGBM_Performance_Graph.png"
    )

)


plt.close()


# ================================================================
# 23. ACTUAL VS PREDICTED
# ================================================================

actual_predicted = pd.DataFrame({

    "Actual": y_test,

    "Predicted": y_pred

})


actual_predicted.to_csv(

    os.path.join(
        OUTPUT_FOLDER,
        "Actual_vs_Predicted.csv"
    ),

    index=False
)


# ================================================================
# 24. FEATURE IMPORTANCE
# ================================================================

feature_importance = pd.DataFrame({

    "Feature": X.columns,

    "Importance": model.feature_importances_

})


feature_importance = feature_importance.sort_values(

    by="Importance",

    ascending=False

)


print(
    "\nTop 10 Important Features:"
)


print(
    feature_importance.head(10)
)


feature_importance.to_csv(

    os.path.join(
        OUTPUT_FOLDER,
        "Feature_Importance.csv"
    ),

    index=False
)


# ================================================================
# 25. FEATURE IMPORTANCE GRAPH
# ================================================================

top_features = feature_importance.head(15)


plt.figure(
    figsize=(10, 7)
)


plt.barh(

    top_features["Feature"][::-1],

    top_features["Importance"][::-1]

)


plt.title(
    "Top 15 LightGBM Feature Importance"
)


plt.xlabel(
    "Importance"
)


plt.ylabel(
    "Feature"
)


plt.tight_layout()


plt.savefig(

    os.path.join(
        OUTPUT_FOLDER,
        "LightGBM_Feature_Importance.png"
    )

)


plt.close()


# ================================================================
# 26. SAVE TEST PREDICTIONS
# ================================================================

test_predictions = X_test.copy()


test_predictions["Actual_Attrition"] = y_test


test_predictions["Predicted_Attrition"] = y_pred


test_predictions.to_csv(

    os.path.join(
        OUTPUT_FOLDER,
        "LightGBM_Test_Predictions.csv"
    ),

    index=False
)


# ================================================================
# 27. SAVE MODEL
# ================================================================

joblib.dump(

    model,

    os.path.join(
        OUTPUT_FOLDER,
        "LightGBM_Attrition_Model.pkl"
    )

)


# ================================================================
# 28. SAVE ENCODERS
# ================================================================

joblib.dump(

    encoders,

    os.path.join(
        OUTPUT_FOLDER,
        "Feature_Encoders.pkl"
    )

)


joblib.dump(

    target_encoder,

    os.path.join(
        OUTPUT_FOLDER,
        "Target_Encoder.pkl"
    )

)


# ================================================================
# 29. SAVE MODEL PARAMETERS
# ================================================================

parameters = pd.DataFrame({

    "Parameter": [
        "n_estimators",
        "learning_rate",
        "max_depth",
        "num_leaves",
        "min_child_samples",
        "subsample",
        "colsample_bytree",
        "scale_pos_weight",
        "random_state"
    ],

    "Value": [
        200,
        0.05,
        5,
        31,
        20,
        0.8,
        0.8,
        scale_pos_weight,
        42
    ]

})


parameters.to_csv(

    os.path.join(
        OUTPUT_FOLDER,
        "LightGBM_Parameters.csv"
    ),

    index=False
)


# ================================================================
# 30. FINAL MESSAGE
# ================================================================

print(
    "\n================================================"
)


print(
    "       LIGHTGBM PROCESS COMPLETED"
)


print(
    "================================================"
)


print(
    "\nAll output files saved in:"
)


print(
    OUTPUT_FOLDER
)
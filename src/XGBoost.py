# ================================================================
# XGBOOST CLASSIFIER - EMPLOYEE ATTRITION PREDICTION
# XGBoost_Classifier_M3.py
# ================================================================
#
# IMPORTANT:
# The original RAW dataset is NEVER modified.
# All preprocessing is performed on copies / inside a pipeline.
#
# OUTPUTS:
#   1. Accuracy
#   2. Precision
#   3. Recall
#   4. F1 Score
#   5. Confusion Matrix
#   6. Performance Graph
#   7. Actual vs Predicted Chart
#   8. Class Distribution Chart
#   9. Feature Importance Chart
#  10. Classification Report
#  11. Test Predictions
#  12. Trained Model
#  13. XGBoost Parameters
# ================================================================


# ================================================================
# 1. IMPORT LIBRARIES
# ================================================================

import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

from xgboost import XGBClassifier


# ================================================================
# 2. RAW DATASET PATH
# ================================================================

# CHANGE THIS PATH ACCORDING TO YOUR COMPUTER

DATASET_PATH = (
    r"C:/Users/AMAN/Documents/Skill_Match/dataset"
    r"\reordered_raw.csv"
)


# ================================================================
# 3. OUTPUT MAIN FOLDER
# ================================================================

OUTPUT_FOLDER = (
    r"C:/Users/AMAN/Documents/Skill_Match/output"
    r"\XGBoost_Classifier_M3_Outputs"
)


# ================================================================
# 4. OUTPUT SUBFOLDERS
# ================================================================

METRICS_FOLDER = os.path.join(
    OUTPUT_FOLDER,
    "metrics"
)

PREDICTIONS_FOLDER = os.path.join(
    OUTPUT_FOLDER,
    "predictions"
)

CONFUSION_FOLDER = os.path.join(
    OUTPUT_FOLDER,
    "confusion_matrix"
)

CHARTS_FOLDER = os.path.join(
    OUTPUT_FOLDER,
    "charts"
)

FEATURE_FOLDER = os.path.join(
    OUTPUT_FOLDER,
    "feature_importance"
)

MODEL_FOLDER = os.path.join(
    OUTPUT_FOLDER,
    "model"
)


# ================================================================
# 5. CREATE FOLDERS ONLY IF THEY DO NOT EXIST
# ================================================================

folders = [
    OUTPUT_FOLDER,
    METRICS_FOLDER,
    PREDICTIONS_FOLDER,
    CONFUSION_FOLDER,
    CHARTS_FOLDER,
    FEATURE_FOLDER,
    MODEL_FOLDER
]


for folder in folders:

    if not os.path.exists(folder):

        os.makedirs(folder)

        print(
            "Created folder:",
            folder
        )

    else:

        print(
            "Folder already exists:",
            folder
        )


# ================================================================
# 6. PROGRAM HEADER
# ================================================================

print("\n")

print("=" * 80)

print(
    "             XGBOOST EMPLOYEE ATTRITION PREDICTION"
)

print("=" * 80)


# ================================================================
# 7. CHECK DATASET
# ================================================================

if not os.path.exists(DATASET_PATH):

    print(
        "\nERROR: Dataset not found."
    )

    print(
        "\nCheck the following path:"
    )

    print(
        DATASET_PATH
    )

    raise SystemExit


print(
    "\nDataset found successfully."
)


# ================================================================
# 8. LOAD RAW DATASET
# ================================================================

df = pd.read_csv(
    DATASET_PATH
)


print(
    "\nRaw dataset loaded successfully."
)


print(
    "Rows    :",
    df.shape[0]
)


print(
    "Columns :",
    df.shape[1]
)


# ================================================================
# 9. CREATE A COPY
# ================================================================

# The original df is NOT modified.

data = df.copy()


# ================================================================
# 10. DISPLAY DATASET INFORMATION
# ================================================================

print(
    "\nDataset columns:"
)


print(
    list(data.columns)
)


print(
    "\nFirst 5 records:"
)


print(
    data.head()
)


# ================================================================
# 11. FIND TARGET COLUMN
# ================================================================

possible_targets = [
    "Attrition",
    "attrition",
    "EmployeeAttrition"
]


target_column = None


for column in possible_targets:

    if column in data.columns:

        target_column = column

        break


# ================================================================
# 12. IF TARGET IS NOT FOUND
# ================================================================

if target_column is None:

    print(
        "\nERROR: Target column could not be detected."
    )

    print(
        "\nAvailable columns:"
    )

    for column in data.columns:

        print(
            column
        )

    print(
        "\nSet target_column manually in the program."
    )

    raise SystemExit


print(
    "\nTarget column:",
    target_column
)


# ================================================================
# 13. REMOVE MISSING TARGET VALUES
# ================================================================

data_model = data.dropna(
    subset=[target_column]
).copy()


print(
    "\nRecords used for modeling:",
    len(data_model)
)


# ================================================================
# 14. SEPARATE FEATURES AND TARGET
# ================================================================

X = data_model.drop(
    columns=[target_column]
).copy()


y = data_model[target_column].copy()


# ================================================================
# 15. REMOVE IDENTIFIER COLUMNS
# ================================================================

possible_id_columns = [
    "EmployeeNumber",
    "EmployeeID",
    "EmployeeId",
    "employee_id",
    "id"
]


id_columns_to_remove = [
    column
    for column in possible_id_columns
    if column in X.columns
]


if len(id_columns_to_remove) > 0:

    print(
        "\nIdentifier columns removed:",
        id_columns_to_remove
    )

    X = X.drop(
        columns=id_columns_to_remove
    )


# ================================================================
# 16. TARGET DISTRIBUTION
# ================================================================

print(
    "\nTarget class distribution:"
)


print(
    y.value_counts()
)


# ================================================================
# 17. ENCODE TARGET
# ================================================================

# XGBoost requires numerical target labels.

target_classes = sorted(
    y.astype(str).unique()
)


target_mapping = {
    label: index
    for index, label in enumerate(target_classes)
}


y = y.astype(str).map(
    target_mapping
)


print(
    "\nTarget encoding:"
)


for label, value in target_mapping.items():

    print(
        label,
        "->",
        value
    )


# ================================================================
# 18. REMOVE COMPLETELY EMPTY FEATURES
# ================================================================

empty_columns = X.columns[
    X.isnull().all()
].tolist()


if len(empty_columns) > 0:

    print(
        "\nCompletely empty columns:"
    )

    print(
        empty_columns
    )

    X = X.drop(
        columns=empty_columns
    )


# ================================================================
# 19. IDENTIFY NUMERIC FEATURES
# ================================================================

numeric_features = X.select_dtypes(
    include=np.number
).columns.tolist()


# ================================================================
# 20. IDENTIFY CATEGORICAL FEATURES
# ================================================================

categorical_features = X.select_dtypes(
    include=[
        "object",
        "category",
        "bool"
    ]
).columns.tolist()


print(
    "\nNumeric features:"
)


print(
    numeric_features
)


print(
    "\nCategorical features:"
)


print(
    categorical_features
)


# ================================================================
# 21. NUMERIC PREPROCESSING
# ================================================================

numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        )
    ]
)


# ================================================================
# 22. CATEGORICAL PREPROCESSING
# ================================================================

categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),

        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)


# ================================================================
# 23. COLUMN TRANSFORMER
# ================================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_transformer,
            numeric_features
        ),

        (
            "categorical",
            categorical_transformer,
            categorical_features
        )
    ],

    remainder="drop"
)


# ================================================================
# 24. TRAIN-TEST SPLIT
# ================================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


print(
    "\nTraining records:",
    len(X_train)
)


print(
    "Testing records :",
    len(X_test)
)


# ================================================================
# 25. CALCULATE CLASS IMBALANCE
# ================================================================

class_counts = y_train.value_counts()


if len(class_counts) == 2:

    negative_class_count = class_counts.get(
        0,
        0
    )

    positive_class_count = class_counts.get(
        1,
        0
    )

    if positive_class_count > 0:

        scale_pos_weight = (
            negative_class_count
            / positive_class_count
        )

    else:

        scale_pos_weight = 1.0

else:

    scale_pos_weight = 1.0


print(
    "\nScale Pos Weight:",
    scale_pos_weight
)


# ================================================================
# 26. CREATE XGBOOST CLASSIFIER
# ================================================================

xgboost_classifier = XGBClassifier(

    n_estimators=200,

    max_depth=5,

    learning_rate=0.05,

    subsample=0.8,

    colsample_bytree=0.8,

    min_child_weight=3,

    gamma=0,

    reg_alpha=0,

    reg_lambda=1,

    objective="binary:logistic",

    eval_metric="logloss",

    scale_pos_weight=scale_pos_weight,

    random_state=42,

    n_jobs=-1
)


# ================================================================
# 27. CREATE MODEL PIPELINE
# ================================================================

model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),

        (
            "classifier",
            xgboost_classifier
        )
    ]
)


# ================================================================
# 28. TRAIN XGBOOST
# ================================================================

print(
    "\nTraining XGBoost..."
)


model.fit(
    X_train,
    y_train
)


print(
    "XGBoost training completed."
)


# ================================================================
# 29. PREDICTION
# ================================================================

print(
    "\nGenerating predictions..."
)


y_pred = model.predict(
    X_test
)


print(
    "Prediction completed."
)


# ================================================================
# 30. CALCULATE ACCURACY
# ================================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)


# ================================================================
# 31. CALCULATE PRECISION
# ================================================================

precision = precision_score(

    y_test,

    y_pred,

    average="weighted",

    zero_division=0
)


# ================================================================
# 32. CALCULATE RECALL
# ================================================================

recall = recall_score(

    y_test,

    y_pred,

    average="weighted",

    zero_division=0
)


# ================================================================
# 33. CALCULATE F1 SCORE
# ================================================================

f1 = f1_score(

    y_test,

    y_pred,

    average="weighted",

    zero_division=0
)


# ================================================================
# 34. DISPLAY PERFORMANCE
# ================================================================

print("\n")

print("=" * 80)

print(
    "                    XGBOOST PERFORMANCE"
)

print("=" * 80)


print(
    f"\nAccuracy  : {accuracy:.4f}"
)


print(
    f"Precision : {precision:.4f}"
)


print(
    f"Recall    : {recall:.4f}"
)


print(
    f"F1 Score  : {f1:.4f}"
)


print(
    "\nPerformance Percentage:"
)


print(
    f"Accuracy  : {accuracy * 100:.2f}%"
)


print(
    f"Precision : {precision * 100:.2f}%"
)


print(
    f"Recall    : {recall * 100:.2f}%"
)


print(
    f"F1 Score  : {f1 * 100:.2f}%"
)


# ================================================================
# 35. SAVE METRICS
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
    ],

    "Percentage": [
        accuracy * 100,
        precision * 100,
        recall * 100,
        f1 * 100
    ]
})


metrics_path = os.path.join(

    METRICS_FOLDER,

    "xgboost_metrics.csv"
)


metrics_df.to_csv(

    metrics_path,

    index=False
)


# ================================================================
# 36. CLASSIFICATION REPORT
# ================================================================

classification_report_result = classification_report(

    y_test,

    y_pred,

    output_dict=True,

    zero_division=0
)


classification_report_df = pd.DataFrame(
    classification_report_result
).transpose()


classification_report_path = os.path.join(

    METRICS_FOLDER,

    "classification_report.csv"
)


classification_report_df.to_csv(

    classification_report_path
)


# ================================================================
# 37. CONFUSION MATRIX
# ================================================================

cm = confusion_matrix(

    y_test,

    y_pred
)


print("\n")


print("=" * 80)


print(
    "                    CONFUSION MATRIX"
)


print("=" * 80)


print(
    cm
)


# ================================================================
# 38. SAVE CONFUSION MATRIX CSV
# ================================================================

class_labels = [
    target_classes[index]
    for index in range(
        len(target_classes)
    )
]


cm_df = pd.DataFrame(

    cm,

    index=[
        "Actual_" + str(label)
        for label in class_labels
    ],

    columns=[
        "Predicted_" + str(label)
        for label in class_labels
    ]
)


cm_csv_path = os.path.join(

    CONFUSION_FOLDER,

    "confusion_matrix.csv"
)


cm_df.to_csv(

    cm_csv_path
)


# ================================================================
# 39. CONFUSION MATRIX GRAPH
# ================================================================

plt.figure(

    figsize=(8, 6)

)


plt.imshow(

    cm

)


plt.title(

    "XGBoost - Attrition Confusion Matrix"

)


plt.xlabel(

    "Predicted Label"

)


plt.ylabel(

    "Actual Label"

)


plt.xticks(

    range(len(class_labels)),

    class_labels,

    rotation=45

)


plt.yticks(

    range(len(class_labels)),

    class_labels

)


# Display matrix values

for i in range(

    cm.shape[0]

):

    for j in range(

        cm.shape[1]

    ):

        plt.text(

            j,

            i,

            str(cm[i, j]),

            ha="center",

            va="center"

        )


plt.colorbar()


plt.tight_layout()


confusion_image_path = os.path.join(

    CONFUSION_FOLDER,

    "confusion_matrix.png"

)


plt.savefig(

    confusion_image_path,

    dpi=300,

    bbox_inches="tight"

)


plt.show()


# ================================================================
# 40. PERFORMANCE GRAPH
# ================================================================

metric_names = [

    "Accuracy",

    "Precision",

    "Recall",

    "F1 Score"

]


metric_values = [

    accuracy * 100,

    precision * 100,

    recall * 100,

    f1 * 100

]


plt.figure(

    figsize=(10, 6)

)


bars = plt.bar(

    metric_names,

    metric_values

)


plt.title(

    "XGBoost Performance - Employee Attrition"

)


plt.xlabel(

    "Evaluation Metric"

)


plt.ylabel(

    "Score (%)"

)


plt.ylim(

    0,

    100

)


# Display metric values

for bar, value in zip(

    bars,

    metric_values

):

    plt.text(

        bar.get_x()
        + bar.get_width() / 2,

        value + 1,

        f"{value:.2f}%",

        ha="center"

    )


plt.tight_layout()


performance_path = os.path.join(

    CHARTS_FOLDER,

    "performance_graph.png"

)


plt.savefig(

    performance_path,

    dpi=300,

    bbox_inches="tight"

)


plt.show()


# ================================================================
# 41. ACTUAL VS PREDICTED CHART
# ================================================================

actual_counts = y_test.value_counts()


predicted_counts = pd.Series(

    y_pred

).value_counts()


comparison_df = pd.DataFrame({

    "Actual": actual_counts,

    "Predicted": predicted_counts

}).fillna(0)


comparison_df = comparison_df.reindex(

    range(len(target_classes))

).fillna(0)


comparison_df.index = class_labels


plt.figure(

    figsize=(9, 6)

)


x = np.arange(

    len(class_labels)

)


width = 0.35


plt.bar(

    x - width / 2,

    comparison_df["Actual"],

    width,

    label="Actual"

)


plt.bar(

    x + width / 2,

    comparison_df["Predicted"],

    width,

    label="Predicted"

)


plt.xlabel(

    "Attrition Class"

)


plt.ylabel(

    "Number of Employees"

)


plt.title(

    "Actual vs Predicted Employee Attrition"

)


plt.xticks(

    x,

    class_labels

)


plt.legend()


plt.tight_layout()


actual_predicted_path = os.path.join(

    CHARTS_FOLDER,

    "actual_vs_predicted.png"

)


plt.savefig(

    actual_predicted_path,

    dpi=300,

    bbox_inches="tight"

)


plt.show()


# ================================================================
# 42. CLASS DISTRIBUTION CHART
# ================================================================

original_class_counts = data_model[
    target_column
].astype(str).value_counts()


plt.figure(

    figsize=(8, 6)

)


plt.bar(

    original_class_counts.index,

    original_class_counts.values

)


plt.xlabel(

    "Attrition Class"

)


plt.ylabel(

    "Number of Employees"

)


plt.title(

    "Employee Attrition Class Distribution"

)


plt.xticks(

    rotation=45

)


plt.tight_layout()


class_distribution_path = os.path.join(

    CHARTS_FOLDER,

    "class_distribution.png"

)


plt.savefig(

    class_distribution_path,

    dpi=300,

    bbox_inches="tight"

)


plt.show()


# ================================================================
# 43. GET FEATURE NAMES AFTER ENCODING
# ================================================================

feature_names = (

    model

    .named_steps["preprocessor"]

    .get_feature_names_out()

)


# ================================================================
# 44. FEATURE IMPORTANCE
# ================================================================

xgb_classifier = (

    model.named_steps["classifier"]

)


feature_importances = (

    xgb_classifier.feature_importances_

)


feature_importance_df = pd.DataFrame({

    "Feature": feature_names,

    "Importance": feature_importances

})


feature_importance_df = (

    feature_importance_df

    .sort_values(

        by="Importance",

        ascending=False

    )

)


# ================================================================
# 45. SAVE FEATURE IMPORTANCE CSV
# ================================================================

feature_importance_path = os.path.join(

    FEATURE_FOLDER,

    "feature_importance.csv"

)


feature_importance_df.to_csv(

    feature_importance_path,

    index=False

)


# ================================================================
# 46. FEATURE IMPORTANCE GRAPH
# ================================================================

top_features = (

    feature_importance_df

    .head(15)

    .sort_values(

        by="Importance"

    )

)


plt.figure(

    figsize=(10, 7)

)


plt.barh(

    top_features["Feature"],

    top_features["Importance"]

)


plt.xlabel(

    "Importance"

)


plt.ylabel(

    "Feature"

)


plt.title(

    "Top 15 XGBoost Feature Importances"

)


plt.tight_layout()


feature_chart_path = os.path.join(

    FEATURE_FOLDER,

    "feature_importance.png"

)


plt.savefig(

    feature_chart_path,

    dpi=300,

    bbox_inches="tight"

)


plt.show()


# ================================================================
# 47. SAVE TEST PREDICTIONS
# ================================================================

test_predictions = X_test.copy()


test_predictions["Actual"] = (

    y_test.map({

        index: label

        for index, label
        in enumerate(target_classes)

    }).values

)


test_predictions["Predicted"] = (

    pd.Series(y_pred)

    .map({

        index: label

        for index, label
        in enumerate(target_classes)

    })

    .values

)


prediction_path = os.path.join(

    PREDICTIONS_FOLDER,

    "test_predictions.csv"

)


test_predictions.to_csv(

    prediction_path,

    index=False

)


# ================================================================
# 48. SAVE TRAINED MODEL
# ================================================================

model_path = os.path.join(

    MODEL_FOLDER,

    "xgboost_model.pkl"

)


with open(

    model_path,

    "wb"

) as file:

    pickle.dump(

        model,

        file

    )


# ================================================================
# 49. SAVE XGBOOST PARAMETERS
# ================================================================

parameters_df = pd.DataFrame({

    "Parameter": [

        "Algorithm",

        "Number of Estimators",

        "Maximum Depth",

        "Learning Rate",

        "Subsample",

        "Column Sample By Tree",

        "Minimum Child Weight",

        "Gamma",

        "Reg Alpha",

        "Reg Lambda",

        "Objective",

        "Evaluation Metric",

        "Scale Pos Weight",

        "Random State"

    ],

    "Value": [

        "XGBoost Classifier",

        xgboost_classifier.n_estimators,

        xgboost_classifier.max_depth,

        xgboost_classifier.learning_rate,

        xgboost_classifier.subsample,

        xgboost_classifier.colsample_bytree,

        xgboost_classifier.min_child_weight,

        xgboost_classifier.gamma,

        xgboost_classifier.reg_alpha,

        xgboost_classifier.reg_lambda,

        xgboost_classifier.objective,

        xgboost_classifier.eval_metric,

        xgboost_classifier.scale_pos_weight,

        xgboost_classifier.random_state

    ]

})


parameters_path = os.path.join(

    METRICS_FOLDER,

    "xgboost_parameters.csv"

)


parameters_df.to_csv(

    parameters_path,

    index=False

)


# ================================================================
# 50. SAVE TARGET MAPPING
# ================================================================

target_mapping_df = pd.DataFrame({

    "Original_Class": target_classes,

    "Encoded_Class": range(

        len(target_classes)

    )

})


target_mapping_path = os.path.join(

    METRICS_FOLDER,

    "target_mapping.csv"

)


target_mapping_df.to_csv(

    target_mapping_path,

    index=False

)


# ================================================================
# 51. FINAL SUMMARY
# ================================================================

print("\n")


print("=" * 80)


print(

    "          XGBOOST ATTRITION PREDICTION COMPLETED"

)


print("=" * 80)


print(

    "\nOriginal RAW dataset was NOT modified."

)


print(

    "\nOriginal dataset:"

)


print(

    DATASET_PATH

)


print(

    "\nAll outputs stored in:"

)


print(

    OUTPUT_FOLDER

)


print("\n")


print(

    "FINAL PERFORMANCE"

)


print(

    "-" * 50

)


print(

    f"Accuracy  : {accuracy * 100:.2f}%"

)


print(

    f"Precision : {precision * 100:.2f}%"

)


print(

    f"Recall    : {recall * 100:.2f}%"

)


print(

    f"F1 Score  : {f1 * 100:.2f}%"

)


print("\n")


print(

    "OUTPUT FOLDERS"

)


print(

    "-" * 50

)


print(

    "1. metrics"

)


print(

    "2. predictions"

)


print(

    "3. confusion_matrix"

)


print(

    "4. charts"

)


print(

    "5. feature_importance"

)


print(

    "6. model"

)


print("\n")


print("=" * 80)


print(

    "                 PROGRAM FINISHED"

)


print("=" * 80)
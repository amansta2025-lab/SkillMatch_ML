# ================================================================
# ADABOOST CLASSIFIER
# EMPLOYEE ATTRITION PREDICTION USING RAW DATASET
# ================================================================
#
# Target:
#     Attrition
#
# No  = Employee stayed
# Yes = Employee left
#
# The original RAW dataset is NEVER modified.
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
#  10. AdaBoost Base Tree Visualization
#  11. Classification Report
#  12. Test Predictions
#  13. Trained Model
#  14. AdaBoost Parameters
#  15. Estimator Weights
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

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

from sklearn.ensemble import AdaBoostClassifier

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


# ================================================================
# 2. RAW DATASET PATH
# ================================================================
#
# CHANGE THIS PATH IF YOUR DATASET IS STORED SOMEWHERE ELSE.
#
# ================================================================

DATASET_PATH = (
    r"C:/Users/AMAN/Documents/Skill_Match/dataset"
    r"\reordered_raw.csv"
)


# ================================================================
# 3. OUTPUT MAIN FOLDER
# ================================================================

OUTPUT_FOLDER = (
    r"C:/Users/AMAN/Documents/Skill_Match/output"
    r"\AdaBoost_Attrition_Outputs"
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

TREE_FOLDER = os.path.join(
    OUTPUT_FOLDER,
    "adaboost_tree"
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
    TREE_FOLDER,
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
    "              ADABOOST EMPLOYEE ATTRITION PREDICTION"
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
# 9. CREATE COPY
# ================================================================
#
# The original dataframe is NEVER modified.
#
# ================================================================

data = df.copy()


# ================================================================
# 10. DISPLAY DATASET
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
# 11. DATASET INFORMATION
# ================================================================

print(
    "\nDataset Information:"
)

print(
    data.info()
)


# ================================================================
# 12. CHECK MISSING VALUES
# ================================================================

print(
    "\nMissing Values:"
)

print(
    data.isnull().sum()
)

print(
    "\nTotal Missing Values:",
    data.isnull().sum().sum()
)


# ================================================================
# 13. CHECK DUPLICATES
# ================================================================

duplicate_count = data.duplicated().sum()

print(
    "\nDuplicate Rows:",
    duplicate_count
)


# ================================================================
# 14. IDENTIFY TARGET COLUMN
# ================================================================

possible_targets = [
    "Attrition",
    "attrition"
]

target_column = None

for column in possible_targets:

    if column in data.columns:

        target_column = column

        break


# ================================================================
# 15. TARGET NOT FOUND
# ================================================================

if target_column is None:

    print(
        "\nERROR: Target column 'Attrition' could not be detected."
    )

    print(
        "\nAvailable columns:"
    )

    for column in data.columns:

        print(
            column
        )

    raise SystemExit


print(
    "\nTarget column:",
    target_column
)


# ================================================================
# 16. CREATE MODELING COPY
# ================================================================

data_model = data.dropna(
    subset=[target_column]
).copy()


print(
    "\nRecords used for modeling:",
    len(data_model)
)


# ================================================================
# 17. SEPARATE FEATURES AND TARGET
# ================================================================

X = data_model.drop(
    columns=[target_column]
).copy()


y = data_model[target_column].copy()


# ================================================================
# 18. CONVERT TARGET TO BINARY
# ================================================================
#
# No  = 0 -> Employee stayed
# Yes = 1 -> Employee left
#
# ================================================================

if y.dtype == "object":

    y = (
        y.astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "no": 0,
                "yes": 1
            }
        )
    )


# ================================================================
# 19. REMOVE INVALID TARGET ROWS
# ================================================================

valid_target = y.notna()

X = X.loc[
    valid_target
].copy()

y = y.loc[
    valid_target
].astype(int).copy()


# ================================================================
# 20. DISPLAY TARGET DISTRIBUTION
# ================================================================

print(
    "\nTarget Class Distribution:"
)

print(
    y.value_counts()
)


print(
    "\nTarget Percentage:"
)

print(
    (
        y.value_counts(
            normalize=True
        )
        * 100
    ).round(2)
)


# ================================================================
# 21. REMOVE COMPLETELY EMPTY FEATURES
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
# 22. REMOVE NON-PREDICTIVE COLUMNS
# ================================================================
#
# EmployeeNumber = employee ID
#
# EmployeeCount
# Over18
# StandardHours
#
# are constant in the standard IBM HR dataset.
#
# ================================================================

unnecessary_columns = [
    "EmployeeNumber",
    "EmployeeCount",
    "Over18",
    "StandardHours"
]


columns_to_remove = [
    column
    for column in unnecessary_columns
    if column in X.columns
]


if len(columns_to_remove) > 0:

    print(
        "\nRemoving non-predictive columns:"
    )

    print(
        columns_to_remove
    )

    X = X.drop(
        columns=columns_to_remove
    )


# ================================================================
# 23. IDENTIFY NUMERICAL FEATURES
# ================================================================

numeric_features = X.select_dtypes(
    include=np.number
).columns.tolist()


# ================================================================
# 24. IDENTIFY CATEGORICAL FEATURES
# ================================================================

categorical_features = X.select_dtypes(
    include=[
        "object",
        "category",
        "bool"
    ]
).columns.tolist()


print(
    "\nNumerical Features:"
)

print(
    numeric_features
)


print(
    "\nCategorical Features:"
)

print(
    categorical_features
)


# ================================================================
# 25. NUMERICAL PREPROCESSING
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
# 26. CATEGORICAL PREPROCESSING
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
# 27. COLUMN TRANSFORMER
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
# 28. TRAIN-TEST SPLIT
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
    "Testing records:",
    len(X_test)
)


# ================================================================
# 29. CREATE BASE DECISION TREE
# ================================================================
#
# AdaBoost uses weak learners.
#
# A shallow Decision Tree works as the weak learner.
#
# ================================================================

base_tree = DecisionTreeClassifier(

    max_depth=2,

    min_samples_split=10,

    min_samples_leaf=5,

    random_state=42
)


# ================================================================
# 30. CREATE ADABOOST CLASSIFIER
# ================================================================

try:

    adaboost = AdaBoostClassifier(

        estimator=base_tree,

        n_estimators=100,

        learning_rate=0.8,

        random_state=42
    )


except TypeError:

    # Compatibility with older scikit-learn versions

    adaboost = AdaBoostClassifier(

        base_estimator=base_tree,

        n_estimators=100,

        learning_rate=0.8,

        random_state=42
    )


# ================================================================
# 31. CREATE COMPLETE MACHINE LEARNING PIPELINE
# ================================================================

model = Pipeline(
    steps=[

        (
            "preprocessor",
            preprocessor
        ),

        (
            "classifier",
            adaboost
        )
    ]
)


# ================================================================
# 32. TRAIN ADABOOST
# ================================================================

print(
    "\nTraining AdaBoost..."
)


model.fit(
    X_train,
    y_train
)


print(
    "AdaBoost training completed."
)


# ================================================================
# 33. MAKE PREDICTIONS
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
# 34. CALCULATE PROBABILITY
# ================================================================

y_probability = model.predict_proba(
    X_test
)[:, 1]


# ================================================================
# 35. CALCULATE ACCURACY
# ================================================================

accuracy = accuracy_score(

    y_test,

    y_pred
)


# ================================================================
# 36. CALCULATE PRECISION
# ================================================================

precision = precision_score(

    y_test,

    y_pred,

    average="weighted",

    zero_division=0
)


# ================================================================
# 37. CALCULATE RECALL
# ================================================================

recall = recall_score(

    y_test,

    y_pred,

    average="weighted",

    zero_division=0
)


# ================================================================
# 38. CALCULATE F1 SCORE
# ================================================================

f1 = f1_score(

    y_test,

    y_pred,

    average="weighted",

    zero_division=0
)


# ================================================================
# 39. DISPLAY MODEL PERFORMANCE
# ================================================================

print("\n")

print("=" * 80)

print(
    "                  ADABOOST PERFORMANCE"
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
# 40. SAVE METRICS
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

    "adaboost_metrics.csv"
)


metrics_df.to_csv(

    metrics_path,

    index=False
)


# ================================================================
# 41. CLASSIFICATION REPORT
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


print(
    "\nClassification Report:"
)


print(

    classification_report(

        y_test,

        y_pred,

        target_names=[

            "Stayed",

            "Left"
        ],

        zero_division=0
    )
)


# ================================================================
# 42. CONFUSION MATRIX
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
# 43. SAVE CONFUSION MATRIX
# ================================================================

cm_df = pd.DataFrame(

    cm,

    index=[

        "Actual_Stayed",

        "Actual_Left"
    ],

    columns=[

        "Predicted_Stayed",

        "Predicted_Left"
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
# 44. CONFUSION MATRIX GRAPH
# ================================================================

plt.figure(

    figsize=(8, 6)
)


plt.imshow(
    cm
)


plt.title(

    "AdaBoost - Employee Attrition Confusion Matrix"
)


plt.xlabel(

    "Predicted Label"
)


plt.ylabel(

    "Actual Label"
)


plt.xticks(

    range(2),

    [
        "Stayed",
        "Left"
    ],

    rotation=45
)


plt.yticks(

    range(2),

    [
        "Stayed",
        "Left"
    ]
)


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
# 45. PERFORMANCE GRAPH
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

    "AdaBoost Employee Attrition Performance"
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
# 46. ACTUAL VS PREDICTED CHART
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

    [0, 1]
).fillna(0)


plt.figure(

    figsize=(9, 6)
)


x = np.arange(

    2
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

    "Actual vs Predicted Employee Attrition - AdaBoost"
)


plt.xticks(

    x,

    [
        "Stayed",
        "Left"
    ]
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
# 47. CLASS DISTRIBUTION
# ================================================================

class_counts = y.value_counts()


plt.figure(

    figsize=(8, 6)
)


plt.bar(

    [
        "Stayed",
        "Left"
    ],

    [
        class_counts.get(0, 0),

        class_counts.get(1, 0)
    ]
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
# 48. GET TRANSFORMED FEATURE NAMES
# ================================================================

feature_names = (

    model

    .named_steps["preprocessor"]

    .get_feature_names_out()
)


# ================================================================
# 49. GET ADABOOST FEATURE IMPORTANCE
# ================================================================

adaboost_classifier = (

    model

    .named_steps["classifier"]
)


feature_importances = (

    adaboost_classifier

    .feature_importances_
)


# ================================================================
# 50. CREATE FEATURE IMPORTANCE DATAFRAME
# ================================================================

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
# 51. SAVE FEATURE IMPORTANCE
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
# 52. DISPLAY TOP FEATURES
# ================================================================

print("\n")

print("=" * 80)

print(
    "                 TOP IMPORTANT FEATURES"
)

print("=" * 80)


print(

    feature_importance_df.head(20)
)


# ================================================================
# 53. FEATURE IMPORTANCE GRAPH
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

    "Top 15 AdaBoost Feature Importances - Employee Attrition"
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
# 54. ADABOOST BASE TREE VISUALIZATION
# ================================================================
#
# AdaBoost consists of multiple weak decision trees.
#
# Here we visualize the FIRST fitted tree.
#
# ================================================================

print("\n")

print("=" * 80)

print(
    "             ADABOOST BASE TREE VISUALIZATION"
)

print("=" * 80)


first_tree = (

    adaboost_classifier

    .estimators_[0]
)


print(

    "\nDisplaying first AdaBoost base decision tree..."
)


plt.figure(

    figsize=(30, 18)
)


plot_tree(

    first_tree,

    feature_names=feature_names,

    class_names=[

        "Stayed",

        "Left"
    ],

    filled=True,

    rounded=True,

    proportion=False,

    precision=2,

    fontsize=8
)


plt.title(

    "AdaBoost - First Base Decision Tree",

    fontsize=20
)


plt.tight_layout()


# ================================================================
# 55. SAVE ADABOOST TREE
# ================================================================

tree_image_path = os.path.join(

    TREE_FOLDER,

    "adaboost_base_tree_1.png"
)


plt.savefig(

    tree_image_path,

    dpi=300,

    bbox_inches="tight"
)


plt.show()


print(

    "\nAdaBoost base tree displayed successfully."
)


print(

    "Tree saved at:"
)


print(

    tree_image_path
)


# ================================================================
# 56. SAVE TEST PREDICTIONS
# ================================================================

test_predictions = X_test.copy()


test_predictions["Actual"] = (

    y_test.values
)


test_predictions["Predicted"] = (

    y_pred
)


# Convert binary values back to Yes/No

test_predictions["Actual_Label"] = (

    test_predictions["Actual"]

    .map({

        0: "No",

        1: "Yes"
    })
)


test_predictions["Predicted_Label"] = (

    test_predictions["Predicted"]

    .map({

        0: "No",

        1: "Yes"
    })
)


test_predictions["Attrition_Probability"] = (

    y_probability
)


# ================================================================
# 57. SAVE PREDICTIONS
# ================================================================

prediction_path = os.path.join(

    PREDICTIONS_FOLDER,

    "test_predictions.csv"
)


test_predictions.to_csv(

    prediction_path,

    index=False
)


# ================================================================
# 58. SAVE TRAINED MODEL
# ================================================================

model_path = os.path.join(

    MODEL_FOLDER,

    "adaboost_attrition_model.pkl"
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
# 59. SAVE ADABOOST PARAMETERS
# ================================================================

parameters_df = pd.DataFrame({

    "Parameter": [

        "Algorithm",

        "Number of Estimators",

        "Learning Rate",

        "Base Tree Maximum Depth",

        "Minimum Samples Split",

        "Minimum Samples Leaf",

        "Random State"
    ],

    "Value": [

        "AdaBoost Classifier",

        adaboost_classifier.n_estimators,

        adaboost_classifier.learning_rate,

        first_tree.max_depth,

        first_tree.min_samples_split,

        first_tree.min_samples_leaf,

        adaboost_classifier.random_state
    ]
})


parameters_path = os.path.join(

    METRICS_FOLDER,

    "adaboost_parameters.csv"
)


parameters_df.to_csv(

    parameters_path,

    index=False
)


# ================================================================
# 60. SAVE ESTIMATOR WEIGHTS
# ================================================================

estimator_weights_df = pd.DataFrame({

    "Estimator_Number": np.arange(

        1,

        len(

            adaboost_classifier.estimator_weights_
        ) + 1
    ),

    "Estimator_Weight": (

        adaboost_classifier

        .estimator_weights_
    )
})


weights_path = os.path.join(

    METRICS_FOLDER,

    "estimator_weights.csv"
)


estimator_weights_df.to_csv(

    weights_path,

    index=False
)


# ================================================================
# 61. FINAL SUMMARY
# ================================================================

print("\n")

print("=" * 80)

print(
    "       ADABOOST EMPLOYEE ATTRITION COMPLETED SUCCESSFULLY"
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
    "5. adaboost_tree"
)


print(
    "6. feature_importance"
)


print(
    "7. model"
)


print("\n")

print("=" * 80)


print(
    "                 PROGRAM FINISHED"
)


print("=" * 80)
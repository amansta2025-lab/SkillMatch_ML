import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Ensure the output directory exists

output_dir = "C:/Users/AMAN/Documents/Skill_Match/output/Corelation_heatmap"
os.makedirs(output_dir, exist_ok=True)


# -------------------------------------------------------------
# 1. Load Employee Attrition Dataset
# -------------------------------------------------------------

df = pd.read_csv(
    "C:/Users/AMAN/Documents/Skill_Match/dataset/reordered_raw.csv"
)

print("Dataset Loaded Successfully. Shape:", df.shape)


# -------------------------------------------------------------
# 2. Compute Correlation Matrix & Generate Heatmap
# -------------------------------------------------------------

# Select only numerical columns for correlation

numerical_cols = df.select_dtypes(
    include=[np.number]
).columns.tolist()

# Remove ID / constant columns from correlation analysis

columns_to_remove = [
    "EmployeeNumber",
    "EmployeeCount",
    "StandardHours"
]

numerical_cols = [
    col for col in numerical_cols
    if col not in columns_to_remove
]

corr_matrix = df[numerical_cols].corr()


# Print matrix to terminal

print("\n--- Correlation Matrix ---")
print(corr_matrix)


# Create Heatmap

plt.figure(figsize=(14, 10))

sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    vmin=-1,
    vmax=1,
    square=True,
    linewidths=0.5
)

plt.title(
    "Correlation Heatmap of Numerical Features",
    fontsize=14,
    fontweight="bold"
)

plt.tight_layout()


# Export Heatmap

heatmap_path = os.path.join(
    output_dir,
    "correlation_heatmap.png"
)

plt.savefig(
    heatmap_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    f"Exported heatmap to: {heatmap_path}"
)


# -------------------------------------------------------------
# 3. Produce Boxplots: Numerical Features vs Attrition
# -------------------------------------------------------------

target_col = "Attrition"


if target_col in df.columns:

    for col in numerical_cols:

        plt.figure(figsize=(6, 5))

        sns.boxplot(
            x=target_col,
            y=col,
            data=df,
            palette="Set2",
            hue=target_col,
            legend=False
        )

        plt.title(
            f"{col} vs {target_col}",
            fontsize=12,
            fontweight="bold"
        )

        plt.xlabel(target_col)
        plt.ylabel(col)

        plt.tight_layout()


        # Export individual boxplot

        boxplot_filename = (
            f"boxplot_{col}_vs_{target_col}.png"
        )

        boxplot_path = os.path.join(
            output_dir,
            boxplot_filename
        )

        plt.savefig(
            boxplot_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print(
            f"Exported boxplot to: {boxplot_path}"
        )


else:

    print(
        f"\nTarget column '{target_col}' "
        "not found in dataset. Skipping boxplots."
    )


print("\nAll EDA tasks completed successfully!")
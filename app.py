import base64
import io
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from flask import Flask, render_template

app = Flask(__name__)

DATA_PATH = os.path.join(app.root_path, "static", "data", "employee_attrition.csv")

# ---- palette shared with style.css ---------------------------------------
INK = "#10231f"
COPPER = "#c1712f"
SLATE = "#3c5b54"
LEAVE = "#b23a2e"
STAY = "#3f7a5b"
LINE = "#d8ddd0"

plt.rcParams.update({
    "font.family": "monospace",
    "axes.edgecolor": LINE,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": "#5a655f",
    "ytick.color": "#5a655f",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def load_data():
    return pd.read_csv(DATA_PATH)


def base_stats(df):
    return {
        "total_employees": len(df),
        "total_features": df.shape[1],
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "departments": df["Department"].nunique(),
        "job_roles": df["JobRole"].nunique(),
        "attrition_count": int((df["Attrition"] == "Yes").sum()),
        "attrition_rate": round((df["Attrition"] == "Yes").mean() * 100, 1),
    }


def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def chart_attrition_split(df):
    counts = df["Attrition"].value_counts()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.pie(
        [counts.get("No", 0), counts.get("Yes", 0)],
        labels=["Stayed", "Left"],
        colors=[STAY, LEAVE],
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
        textprops={"fontsize": 10},
    )
    ax.set_title("Attrition Split", fontsize=12, color=INK, fontweight="bold")
    return fig_to_b64(fig)


def chart_department(df):
    grp = df.groupby("Department")["Attrition"].apply(lambda s: (s == "Yes").mean() * 100)
    grp = grp.sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(grp.index, grp.values, color=COPPER, width=0.55)
    ax.set_ylabel("Attrition rate (%)")
    ax.set_title("Attrition Rate by Department", fontsize=12, color=INK, fontweight="bold")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    plt.xticks(rotation=12, ha="right", fontsize=9)
    return fig_to_b64(fig)


def chart_overtime(df):
    grp = df.groupby("OverTime")["Attrition"].apply(lambda s: (s == "Yes").mean() * 100)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(grp.index, grp.values, color=[SLATE, LEAVE], width=0.45)
    ax.set_ylabel("Attrition rate (%)")
    ax.set_title("Attrition Rate by Overtime", fontsize=12, color=INK, fontweight="bold")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    return fig_to_b64(fig)


def chart_income(df):
    stay = df.loc[df["Attrition"] == "No", "MonthlyIncome"]
    leave = df.loc[df["Attrition"] == "Yes", "MonthlyIncome"]
    fig, ax = plt.subplots(figsize=(5, 4))
    bp = ax.boxplot([stay, leave], tick_labels=["Stayed", "Left"], patch_artist=True, widths=0.5)
    for patch, color in zip(bp["boxes"], [STAY, LEAVE]):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.set_ylabel("Monthly income ($)")
    ax.set_title("Monthly Income vs Attrition", fontsize=12, color=INK, fontweight="bold")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    return fig_to_b64(fig)


@app.route("/")
def home():
    df = load_data()
    return render_template("home.html", active="home", **base_stats(df))


@app.route("/about")
def about():
    df = load_data()
    return render_template("about.html", active="about", **base_stats(df))


@app.route("/dataset")
def dataset():
    df = load_data()
    stats = base_stats(df)
    sample_rows = df[["Age", "Department", "JobRole", "OverTime", "MonthlyIncome", "YearsAtCompany", "Attrition"]].head(8).to_dict("records")
    return render_template("dataset.html", active="dataset", sample_rows=sample_rows, **stats)


@app.route("/preprocessing")
def preprocessing():
    df = load_data()
    return render_template("preprocessing.html", active="preprocessing", **base_stats(df))


@app.route("/visualization")
def visualization():
    df = load_data()
    stats = base_stats(df)
    charts = {
        "split": chart_attrition_split(df),
        "department": chart_department(df),
        "overtime": chart_overtime(df),
        "income": chart_income(df),
    }
    return render_template("visualization.html", active="visualization", charts=charts, **stats)


@app.route("/models")
def models():
    df = load_data()
    return render_template("models.html", active="models", **base_stats(df))


@app.route("/prediction")
def prediction():
    df = load_data()
    stats = base_stats(df)
    form_options = {
        "dept_options": sorted(df["Department"].unique().tolist()),
        "role_options": sorted(df["JobRole"].unique().tolist()),
        "education_fields": sorted(df["EducationField"].unique().tolist()),
        "marital_statuses": sorted(df["MaritalStatus"].unique().tolist()),
        "travel_options": sorted(df["BusinessTravel"].unique().tolist()),
    }
    return render_template("prediction.html", active="prediction", **form_options, **stats)


@app.route("/dashboard")
def dashboard():
    df = load_data()
    stats = base_stats(df)
    dept_breakdown = (
        df.groupby("Department")
        .agg(total=("Attrition", "size"), left=("Attrition", lambda s: (s == "Yes").sum()))
        .reset_index()
    )
    dept_breakdown["rate"] = (dept_breakdown["left"] / dept_breakdown["total"] * 100).round(1)
    dept_breakdown = dept_breakdown.sort_values("rate", ascending=False).to_dict("records")
    chart = chart_department(df)
    return render_template("dashboard.html", active="dashboard", dept_breakdown=dept_breakdown, chart=chart, **stats)


@app.route("/reports")
def reports():
    df = load_data()
    return render_template("reports.html", active="reports", **base_stats(df))


@app.route("/contact")
def contact():
    df = load_data()
    return render_template("contact.html", active="contact", **base_stats(df))


if __name__ == "__main__":
    app.run(debug=True)

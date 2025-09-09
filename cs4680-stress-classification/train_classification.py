"""
Train classification models to predict stress level (Low/Moderate/High).
"""
# Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from pathlib import Path

#Kaggle Dataset Path
DATA_PATH = "academic Stress level - maintainance 1.csv"
RANDOM_STATE = 42

#Assigning Labels to Values
def map_to_label(x):
    if pd.isna(x): return np.nan
    if x <= 2: return "Low"
    if x == 3: return "Moderate"
    return "High"

def main():
    # Load and Clean the Dataset
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip() for c in df.columns]
    rating_col = "Rate your academic stress index"
    df[rating_col] = pd.to_numeric(df[rating_col], errors="coerce")
    df["stress_level"] = df[rating_col].apply(map_to_label)

    X = df.drop(columns=["Timestamp", rating_col, "stress_level"])
    y = df["stress_level"]

    # Split Columns by Type
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    # Preprocessing Pipeline
    preprocessor = ColumnTransformer([
        ("num", Pipeline([("scaler", StandardScaler())]), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
    ])

    # Define Models
    models = {
        "LogisticRegression": LogisticRegression(max_iter=2000, multi_class="multinomial", random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE),
    }

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # Cross-Validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    results = {}
    conf_mats = {}

    # Train/Evaluate Models
    for name, model in models.items():
        pipe = Pipeline([("preprocess", preprocessor), ("model", model)])
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1_macro")
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        acc = accuracy_score(y_test, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, preds, average="macro", zero_division=0)

        # Record Summary Metrics
        results[name] = {
            "cv_f1_macro_mean": float(np.mean(scores)),
            "cv_f1_macro_std": float(np.std(scores)),
            "accuracy": float(acc),
            "precision_macro": float(precision),
            "recall_macro": float(recall),
            "f1_macro": float(f1),
        }

        # Confusion Matrix
        conf_mats[name] = confusion_matrix(y_test, preds, labels=["Low","Moderate","High"])

        # Console Summary for Quick Inspection
        print(f"\n=== {name} ===")
        print("CV F1_macro: %.3f ± %.3f" % (np.mean(scores), np.std(scores)))
        print("Test accuracy: %.3f" % acc)
        print("Test macro F1: %.3f" % f1)
        print(classification_report(y_test, preds, zero_division=0))

        # Plot CM
        Path("plots").mkdir(exist_ok=True, parents=True)
        fig = plt.figure()
        plt.imshow(conf_mats[name], interpolation="nearest")
        plt.title(f"Confusion Matrix - {name}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        tick_marks = np.arange(3)
        plt.xticks(tick_marks, ["Low", "Moderate", "High"], rotation=45)
        plt.yticks(tick_marks, ["Low", "Moderate", "High"])
        for i in range(3):
            for j in range(3):
                plt.text(j, i, conf_mats[name][i,j], ha="center", va="center")
        plt.tight_layout()
        plt.savefig(f"plots/confusion_{name}.png", bbox_inches="tight")
        plt.close(fig)

    # Save Predictions of the Best Model
    best_name = max(results.items(), key=lambda kv: kv[1]["f1_macro"])[0]
    best_pipe = Pipeline([("preprocess", preprocessor), ("model", models[best_name])])
    best_pipe.fit(X_train, y_train)
    preds = best_pipe.predict(X_test)
    out = X_test.copy()
    out["true"] = y_test.values
    out["pred"] = preds
    Path("artifacts").mkdir(exist_ok=True, parents=True)
    out.to_csv("artifacts/test_predictions.csv", index=False)

if __name__ == "__main__":
    main()

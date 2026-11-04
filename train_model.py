"""Preprocess data, train a classifier, evaluate it, and save model artifacts."""
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

DATA = Path("data/student_performance.csv")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")

def main():
    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(DATA)
    target = "performance"
    X, y = df.drop(columns=[target]), df[target]
    numeric = ["attendance", "study_hours", "previous_score",
               "assignments_completed", "sleep_hours"]
    categorical = ["internet_access", "extracurricular"]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    prep = ColumnTransformer([
        ("numeric", numeric_pipe, numeric),
        ("categorical", categorical_pipe, categorical)
    ])
    model = Pipeline([
        ("preprocessing", prep),
        ("classifier", RandomForestClassifier(
            n_estimators=250, random_state=42, class_weight="balanced"
        ))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "macro_precision": round(float(precision_score(y_test, pred, average="macro", zero_division=0)), 4),
        "macro_recall": round(float(recall_score(y_test, pred, average="macro", zero_division=0)), 4),
        "macro_f1": round(float(f1_score(y_test, pred, average="macro", zero_division=0)), 4),
        "test_rows": int(len(y_test)),
        "classification_report": classification_report(y_test, pred, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, pred, labels=["Low", "Medium", "High"]).tolist(),
        "class_order": ["Low", "Medium", "High"]
    }
    joblib.dump(model, MODEL_DIR/"student_performance_model.joblib")
    (REPORT_DIR/"metrics.json").write_text(json.dumps(metrics, indent=2))
    print("Model saved to models/student_performance_model.joblib")
    print(json.dumps({k:v for k,v in metrics.items() if k in ["accuracy","macro_precision","macro_recall","macro_f1"]}, indent=2))

if __name__ == "__main__":
    main()

# ============================================================
# STUDENT PERFORMANCE PREDICTION SYSTEM
# train_model.py
# Downloads the UCI dataset, prepares the data, trains and
# evaluates a Random Forest model, then saves the model.
# ============================================================

import pandas as pd
import joblib
from ucimlrepo import fetch_ucirepo

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

DATA_FILE = "student_data.csv"
MODEL_FILE = "student_model.pkl"

def main():
    print("=" * 60)
    print("STUDENT PERFORMANCE PREDICTION SYSTEM")
    print("=" * 60)

    # ACTIVITY A: Retrieve and inspect the UCI dataset
    print("\n[1/4] Downloading UCI Student Performance dataset...")
    student = fetch_ucirepo(id=320)

    # Combine predictor columns and grade target
    df = pd.concat(
        [student.data.features, student.data.targets],
        axis=1
    )

    # The laboratory specifies the Mathematics records.
    # The UCI repository's student performance data includes
    # school-subject records; this selection keeps the math file
    # when the repository provides a subject indicator.
    if "subject" in df.columns:
        math_rows = df["subject"].astype(str).str.lower().isin(
            ["math", "mathematics", "mat"]
        )
        if math_rows.any():
            df = df.loc[math_rows].copy()

    if "G3" not in df.columns:
        raise ValueError(
            "The downloaded dataset does not contain the required G3 column."
        )

    df.to_csv(DATA_FILE, index=False)
    print(f"Dataset saved as {DATA_FILE}")
    print("Rows and columns:", df.shape)
    print("\nFirst five records:")
    print(df.head())
    print("\nColumn data types:")
    print(df.dtypes)
    print("\nMissing values per column:")
    print(df.isnull().sum())

    # ACTIVITY B: Preprocess, train, and evaluate
    print("\n[2/4] Preparing data...")
    df = df.drop_duplicates().copy()

    # Target: 1 = at risk (G3 < 10); 0 = not at risk
    y = (df["G3"] < 10).astype(int)

    # Never use G3 as an input feature (prevents target leakage)
    X = df.drop(columns=["G3"])

    if y.nunique() < 2:
        raise ValueError(
            "The target contains only one class. A classifier needs "
            "both at-risk and not-at-risk records."
        )

    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = X.select_dtypes(exclude="number").columns.tolist()

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    transformers = []
    if numeric_cols:
        transformers.append(("numeric", numeric_pipe, numeric_cols))
    if categorical_cols:
        transformers.append(("categorical", categorical_pipe, categorical_cols))

    preprocessor = ColumnTransformer(transformers=transformers)

    # Stratification keeps the class proportions similar in both sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight="balanced"
        ))
    ])

    print("\n[3/4] Training Random Forest model...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    print("\nMODEL EVALUATION")
    print("-" * 40)
    print("Accuracy:", round(accuracy, 3))
    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Not at risk", "At risk"],
        zero_division=0
    ))
    print("Confusion matrix (labels: 0=Not at risk, 1=At risk):")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))

    # Save the complete preprocessing + model pipeline.
    joblib.dump(model, MODEL_FILE)
    print(f"\n[4/4] Trained model saved as {MODEL_FILE}")
    print("\nCompleted. You can now run: streamlit run app.py")

if __name__ == "__main__":
    main()

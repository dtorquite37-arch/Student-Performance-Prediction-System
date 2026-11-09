# ============================================================
# STUDENT PERFORMANCE PREDICTION SYSTEM
# app.py
# Streamlit interface for uploading student records and
# predicting whether each record is AT RISK or NOT AT RISK.
# ============================================================

import os
import streamlit as st
import pandas as pd
import joblib

DATA_FILE = "student_data.csv"
MODEL_FILE = "student_model.pkl"

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Student Performance Prediction System")
st.caption(
    "Classroom prototype for exploring possible academic risk "
    "using historical student records."
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)

@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_FILE)

# Check that the training script has been run first.
if not os.path.exists(MODEL_FILE) or not os.path.exists(DATA_FILE):
    st.error(
        "Required files are missing. Run `python train_model.py` "
        "in the project folder first. This creates student_data.csv "
        "and student_model.pkl."
    )
    st.stop()

try:
    model = load_model()
    data = load_dataset()
except Exception as error:
    st.error(f"Unable to load the model or dataset: {error}")
    st.stop()

if "G3" not in data.columns:
    st.error("The dataset does not contain the required G3 column.")
    st.stop()

# Dashboard summary
c1, c2, c3 = st.columns(3)
c1.metric("Total student records", f"{len(data):,}")
c2.metric("Average final grade (G3)", f"{data['G3'].mean():.2f}")
c3.metric("Historical at-risk records", f"{int((data['G3'] < 10).sum()):,}")

st.divider()

st.subheader("📊 Dataset preview")
st.write("The table below shows the first 10 records from the downloaded dataset.")
st.dataframe(data.head(10), use_container_width=True)

st.divider()

st.subheader("🔎 Predict uploaded student records")
st.write(
    "Upload a CSV file containing the same predictor columns used during "
    "training. Do not include G3; it is the target grade, not an input."
)

# Provide a sample CSV template with predictor columns only.
template = data.drop(columns=["G3"]).head(5)
st.download_button(
    label="⬇️ Download sample input CSV",
    data=template.to_csv(index=False).encode("utf-8"),
    file_name="sample_prediction_input.csv",
    mime="text/csv"
)

uploaded_file = st.file_uploader(
    "Choose student CSV file",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        new_data = pd.read_csv(uploaded_file)

        if new_data.empty:
            st.error("The uploaded CSV file is empty.")
            st.stop()

        required_columns = list(model.feature_names_in_)
        missing_columns = [
            column for column in required_columns
            if column not in new_data.columns
        ]

        if missing_columns:
            st.error(
                "Missing required columns: "
                + ", ".join(missing_columns)
            )
        else:
            # Ignore extra columns and arrange predictors in training order.
            prediction_input = new_data[required_columns].copy()
            predictions = model.predict(prediction_input)

            results = new_data.copy()
            results["Prediction"] = [
                "AT RISK" if value == 1 else "NOT AT RISK"
                for value in predictions
            ]

            st.success("Prediction completed successfully.")
            st.subheader("📋 Prediction results")
            st.dataframe(results, use_container_width=True)

            at_risk_count = int((predictions == 1).sum())
            not_at_risk_count = int((predictions == 0).sum())

            m1, m2 = st.columns(2)
            m1.metric("Predicted AT RISK", at_risk_count)
            m2.metric("Predicted NOT AT RISK", not_at_risk_count)

            st.download_button(
                label="⬇️ Download prediction results",
                data=results.to_csv(index=False).encode("utf-8"),
                file_name="student_predictions.csv",
                mime="text/csv"
            )

    except pd.errors.EmptyDataError:
        st.error("The uploaded file does not contain readable CSV data.")
    except Exception as error:
        st.error(f"An error occurred while processing the file: {error}")

st.divider()
st.warning(
    "Responsible use: This is a classroom demonstration using public, "
    "historical data. Predictions are estimates, not diagnoses, and must "
    "not be used as the sole basis for real student decisions."
)

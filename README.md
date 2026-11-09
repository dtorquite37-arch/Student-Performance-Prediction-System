# 🎓 Student Performance Prediction System

A classroom prototype (Data Mining laboratory) that estimates whether a student is at risk of failing the final Mathematics grade.

## Problem
Academic risk is often noticed only after poor results. This prototype explores early identification of learners who may need additional academic support, using indicators such as study time, absences, past failures, and the first two period grades (G1, G2).

## Dataset
UCI Machine Learning Repository — [Student Performance](https://archive.ics.uci.edu/dataset/320/student+performance) (Mathematics file, `student-mat.csv`, 395 records, Portuguese secondary schools).

## Prediction target
`G3 < 10` → **AT RISK** (1); `G3 ≥ 10` → **NOT AT RISK** (0).
`G3` is the target only and is never used as an input feature (prevents target leakage).

## Technologies
Python 3.10+, pandas, scikit-learn (Random Forest pipeline), joblib, Streamlit, Git/GitHub.

## Installation and run
```bash
pip install -r requirements.txt
python train_model.py      # downloads data, trains, evaluates, saves student_model.pkl
streamlit run app.py       # opens http://localhost:8501
```
In the app, download the sample input CSV, upload it, review predictions, and download the results.

## Files
| File | Purpose |
|---|---|
| `train_model.py` | Retrieves data, preprocesses, trains and evaluates the model |
| `app.py` | Streamlit interface: dataset summary, CSV upload, validation, predictions, export |
| `requirements.txt` | Python dependencies |

## Limitations and responsible use
- Trained on one Portuguese secondary-school dataset; it may not generalize to other institutions or populations.
- G1 and G2 are highly correlated with G3, so accuracy is optimistic; the model is less useful for very early prediction.
- Small dataset (395 records) → the test set is small and metrics vary.
- Predictions are **not diagnoses** and must not be the sole basis for real student decisions. They should prompt a supportive conversation, not a label.

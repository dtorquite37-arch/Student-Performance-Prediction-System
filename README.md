# Student Performance Prediction System

## Project overview
The Student Performance Prediction System is a classroom prototype that uses historical student information to estimate whether a student may be at risk of failing the final Mathematics grade.

**Prediction rule**
- `G3 < 10` = **AT RISK**
- `G3 >= 10` = **NOT AT RISK**

`G3` is the target only. It is excluded from the model input features to prevent target leakage.

## SAD mini-analysis

| SAD element | Student response |
|---|---|
| Problem statement | Academic risk is often noticed after poor results. This prototype explores early identification using available student indicators. |
| Users | Faculty member or academic analyst. |
| Inputs | Student attributes and academic indicators such as study time, absences, failures, G1 and G2. |
| Process | Validate data → preprocess → apply trained model → classify risk. |
| Outputs | AT RISK / NOT AT RISK label and model evaluation summary. |
| Key limitation | A model trained on one Portuguese secondary-school dataset may not generalize to another institution or population. |

**Mini use-case:** The user uploads a CSV file → the system checks required columns → the model predicts risk → the system displays and exports the results.

## Dataset source
UCI Machine Learning Repository — Student Performance:
https://archive.ics.uci.edu/dataset/320/student+performance

The training script retrieves the dataset directly using `ucimlrepo` and creates `student_data.csv`.

## Technologies
- Python
- pandas
- scikit-learn
- Random Forest Classifier
- Streamlit
- GitHub
- joblib
- ucimlrepo

## Project files
- `train_model.py` — downloads the dataset, preprocesses the data, trains and evaluates the model, and saves the model.
- `app.py` — Streamlit application for viewing dataset information, uploading CSV records, predicting risk, and downloading results.
- `requirements.txt` — required Python packages.
- `README.md` — project documentation.

Generated after running the training script:
- `student_data.csv`
- `student_model.pkl`

## Installation and running

1. Install Python 3.10 or later.
2. Open a terminal in this project folder.
3. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

4. Download the dataset, train the model, and generate the required files:

   ```bash
   python train_model.py
   ```

5. Start the Streamlit application:

   ```bash
   streamlit run app.py
   ```

6. Open the local address displayed in the terminal, normally:

   `http://localhost:8501`

7. Download the sample input CSV from the app, then upload it to test predictions.

## Model evaluation
When `train_model.py` runs, it prints:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

Record the actual values displayed in your terminal in your laboratory report. Do not invent or assume evaluation scores. Accuracy should not be interpreted alone.

## Testing checklist
- [ ] Run `train_model.py`; confirm `student_data.csv` and `student_model.pkl` are created.
- [ ] Run the Streamlit app; confirm the dashboard opens.
- [ ] Upload the sample CSV; confirm predictions appear.
- [ ] Upload a CSV missing a required column; confirm a helpful error appears.
- [ ] Download the prediction results CSV.
- [ ] Open the GitHub repository and confirm the source files and README are visible.

## Reflection questions and answers

**1. What is the prediction target, and why is G3 excluded from the input features?**

The prediction target is whether a student is at risk of failing the final Mathematics grade. A G3 below 10 means AT RISK, while a G3 of 10 or higher means NOT AT RISK. G3 is excluded because it is the final grade being predicted. Including it as an input would reveal the answer to the model and cause target leakage.

**2. What does the model's recall for the 'At risk' class tell you?**

Recall for the At risk class tells us how many of the students who are actually at risk were correctly identified by the model. A higher recall means fewer at-risk students were missed.

**3. Why might a model trained on this dataset perform differently on students from your institution?**

The dataset comes from a particular Portuguese secondary-school population. Students in another institution may have different backgrounds, learning conditions, grading practices, and academic experiences. Because of these differences, the model may not make equally accurate predictions.

**4. What action should a faculty member take after seeing an 'AT RISK' prediction?**

The faculty member should review the student's actual situation, talk with the student, check other academic information, and consider providing appropriate support. The prediction should not be the only basis for a decision.

## Responsible-use limitation
This is a classroom demonstration using public, historical data. Predictions are not diagnoses and must not be used as the sole basis for real student decisions.

## GitHub publishing

Create a public repository named `student-performance-prediction`. Then run these commands in the project folder, replacing `YOUR-USERNAME` with your GitHub username:

```bash
git init
git add train_model.py app.py requirements.txt README.md
git commit -m "Build student performance prediction prototype"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/student-performance-prediction.git
git push -u origin main
```

Do not put GitHub passwords or personal access tokens in source code.

## References
- UCI Student Performance Dataset: https://archive.ics.uci.edu/dataset/320/student+performance
- Scikit-learn documentation: https://scikit-learn.org/stable/
- Streamlit documentation: https://docs.streamlit.io/
- GitHub getting started: https://docs.github.com/en/get-started

# Student Performance Prediction Using Machine Learning

## 1. Project overview
This project demonstrates a machine-learning workflow that estimates a student's performance category (**Low**, **Medium**, or **High**) from learning-related inputs. It includes data preprocessing, exploratory data analysis, model training/evaluation, a Flask web application, and a Streamlit interface.

> **Important data note:** `data/student_performance.csv` is a **synthetic teaching dataset generated for this project**. It is not real student data and does not establish real-world predictive accuracy. Do not use the prediction to grade, rank, punish, or make high-impact decisions about students.

## 2. Rubric coverage
| Rubric item | Project implementation |
|---|---|
| Problem understanding (10) | Problem statement, objective, inputs and output documented below |
| Preprocessing & EDA (10) | Missing-value imputation, scaling, one-hot encoding, dataset summary and Streamlit chart |
| ML development (20) | Random Forest classifier in a reproducible scikit-learn pipeline |
| Evaluation & critical analysis (20) | Stratified train/test split; accuracy, macro precision/recall/F1, classification report and confusion matrix |
| Flask & Streamlit (25) | `app.py` and `streamlit_app.py` |
| Visualization & GitHub (15) | Streamlit class distribution chart; repository-ready documentation |

## 3. Problem statement
Students' academic outcomes can be associated with several learning indicators. The project demonstrates how structured inputs can be processed by a classification model to estimate a broad performance category.

## 4. Objectives
- Prepare a structured dataset for machine learning.
- Explore the distribution of performance categories.
- Preprocess numerical and categorical features.
- Train a classification model.
- Evaluate the model using multiple metrics.
- Present predictions through Flask and Streamlit.

## 5. Features
- Attendance percentage
- Study hours per day
- Previous exam score
- Assignments completed (out of 10)
- Sleep hours per day
- Internet access (Yes/No)
- Extracurricular participation (Yes/No)

Target: `performance` with Low, Medium, and High classes.

## 6. Technology stack
Python, Pandas, NumPy, scikit-learn, Joblib, Flask, Streamlit.

## 7. Setup
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
```

## 8. Train the model
The CSV is already included. To regenerate it:
```bash
python generate_data.py
```
Train and evaluate:
```bash
python train_model.py
```
The trained pipeline is saved to `models/student_performance_model.joblib` and evaluation details to `reports/metrics.json`.

## 9. Run Flask
```bash
python app.py
```
Open `http://127.0.0.1:5000`.

## 10. Run Streamlit
```bash
streamlit run streamlit_app.py
```

## 11. Evaluation
Run `python train_model.py` to generate metrics for the current environment and dataset split. The reported metrics are for a synthetic dataset and must not be interpreted as evidence of performance on real students. Macro-averaged precision, recall and F1 are included so each class contributes equally to the summary.

## 12. Critical analysis and limitations
- Synthetic data may encode artificial relationships and does not represent the diversity of real classrooms.
- A random train/test split estimates performance only on similar generated records.
- Features such as sleep, internet access and extracurricular participation can reflect circumstances outside a student's control.
- Predictions can be wrong and may reproduce bias if trained on unrepresentative data.
- A real deployment would require ethically collected, consented, representative data; privacy safeguards; stakeholder review; bias testing; external validation; and human oversight.
- The model should be used only as a classroom demonstration or to prompt supportive discussion, never as the sole basis for educational decisions.

## 13. Future enhancement
Use a properly governed real dataset, compare multiple algorithms, add cross-validation and calibration, include explainability, and conduct subgroup fairness checks before any real-world pilot.

## 14. Project structure
```text
student_performance_prediction_ml/
├── app.py
├── streamlit_app.py
├── train_model.py
├── generate_data.py
├── requirements.txt
├── README.md
├── PROJECT_REPORT.md
├── data/student_performance.csv
├── models/student_performance_model.joblib
├── reports/metrics.json
└── templates/index.html
```

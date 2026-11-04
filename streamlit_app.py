"""Streamlit dashboard for interactive student performance prediction."""
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="centered")
st.title("🎓 Student Performance Prediction")
st.write("Enter student learning details to estimate the performance category.")
model_path = Path("models/student_performance_model.joblib")
if not model_path.exists():
    st.error("Model not found. First run: python train_model.py")
    st.stop()
model = joblib.load(model_path)

with st.form("prediction_form"):
    attendance = st.slider("Attendance (%)", 0, 100, 80)
    study_hours = st.slider("Study hours per day", 0.0, 12.0, 3.0, 0.5)
    previous_score = st.slider("Previous exam score", 0, 100, 65)
    assignments = st.slider("Assignments completed (out of 10)", 0, 10, 7)
    sleep_hours = st.slider("Sleep hours per day", 0.0, 12.0, 7.0, 0.5)
    internet = st.selectbox("Internet access", ["Yes", "No"])
    extra = st.selectbox("Extracurricular participation", ["Yes", "No"])
    submitted = st.form_submit_button("Predict performance")

if submitted:
    row = pd.DataFrame([{
        "attendance": attendance, "study_hours": study_hours,
        "previous_score": previous_score, "assignments_completed": assignments,
        "sleep_hours": sleep_hours, "internet_access": internet,
        "extracurricular": extra
    }])
    result = model.predict(row)[0]
    st.subheader(f"Predicted category: {result}")
    st.info("This is an educational demonstration, not a final judgement of a student's ability.")

st.divider()
st.subheader("Dataset overview")
data = pd.read_csv("data/student_performance.csv")
st.write(f"Rows: {len(data)} | Features: {len(data.columns)-1}")
st.bar_chart(data["performance"].value_counts())

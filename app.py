"""Flask web application for student performance prediction."""
from pathlib import Path
import joblib
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)
MODEL_PATH = Path("models/student_performance_model.joblib")
model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    error = None
    if request.method == "POST":
        try:
            row = pd.DataFrame([{
                "attendance": float(request.form["attendance"]),
                "study_hours": float(request.form["study_hours"]),
                "previous_score": float(request.form["previous_score"]),
                "assignments_completed": float(request.form["assignments_completed"]),
                "sleep_hours": float(request.form["sleep_hours"]),
                "internet_access": request.form["internet_access"],
                "extracurricular": request.form["extracurricular"]
            }])
            if model is None:
                error = "Model file not found. Run: python train_model.py"
            else:
                prediction = model.predict(row)[0]
        except Exception as exc:
            error = f"Please check the entered values. ({exc})"
    return render_template("index.html", prediction=prediction, error=error)

if __name__ == "__main__":
    app.run(debug=True)

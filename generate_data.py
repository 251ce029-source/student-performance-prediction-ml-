"""Generate the synthetic student performance dataset used for this project."""
from pathlib import Path
import numpy as np
import pandas as pd

def main():
    rng = np.random.default_rng(42)
    n = 800
    attendance = rng.integers(45, 101, n)
    study_hours = np.round(rng.uniform(0.5, 8.0, n), 1)
    previous_score = rng.integers(30, 101, n)
    assignments = rng.integers(0, 11, n)
    sleep_hours = np.round(rng.uniform(4.0, 9.5, n), 1)
    internet = rng.choice(["Yes", "No"], n, p=[0.82, 0.18])
    extra = rng.choice(["Yes", "No"], n, p=[0.55, 0.45])
    noise = rng.normal(0, 9, n)
    score = (0.28*previous_score + 0.28*attendance + 4*study_hours +
             1.5*assignments + 1.3*sleep_hours +
             2*(internet=="Yes") + (extra=="Yes") + noise - 25)
    score = np.clip(score, 0, 100)
    label = np.where(score >= 70, "High", np.where(score >= 50, "Medium", "Low"))
    data = pd.DataFrame({
        "attendance": attendance, "study_hours": study_hours,
        "previous_score": previous_score, "assignments_completed": assignments,
        "sleep_hours": sleep_hours, "internet_access": internet,
        "extracurricular": extra, "performance": label
    })
    out = Path("data")
    out.mkdir(exist_ok=True)
    data.to_csv(out/"student_performance.csv", index=False)
    print(f"Saved {len(data)} rows to data/student_performance.csv")

if __name__ == "__main__":
    main()

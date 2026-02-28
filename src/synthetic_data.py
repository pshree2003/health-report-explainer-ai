from __future__ import annotations

import numpy as np
import pandas as pd


def generate_synthetic_dataset(n: int = 500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 80, n)
    gender = rng.choice(["Male", "Female"], n)

    hemoglobin = rng.normal(13.2, 1.4, n) - (gender == "Female") * 0.7
    wbc = rng.normal(7.0, 2.0, n)
    rbc = rng.normal(4.8, 0.6, n)
    platelets = rng.normal(280, 60, n)
    cholesterol = rng.normal(190, 35, n) + (age > 45) * 15
    hdl = rng.normal(50, 12, n)
    ldl = rng.normal(120, 30, n) + (age > 45) * 12
    triglycerides = rng.normal(140, 45, n)

    symptoms = np.where(hemoglobin < 11.5, "Fatigue", np.where(wbc > 11, "Fever", "None"))

    df = pd.DataFrame(
        {
            "Patient_ID": [f"P-{1000+i}" for i in range(n)],
            "Test_Date": pd.Timestamp("today").normalize() - pd.to_timedelta(rng.integers(0, 365, n), unit="D"),
            "Hemoglobin": hemoglobin.round(1),
            "WBC": wbc.round(1),
            "RBC": rbc.round(2),
            "Platelets": platelets.round(0),
            "Cholesterol": cholesterol.round(0),
            "HDL": hdl.round(0),
            "LDL": ldl.round(0),
            "Triglycerides": triglycerides.round(0),
            "Age": age,
            "Gender": gender,
            "Symptoms": symptoms,
        }
    )
    return df

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_synthetic_dataset(n: int = 500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    patient_pool = max(25, n // 5)
    patient_ids = np.array([f"P-{1000 + i}" for i in range(patient_pool)])
    assigned_patients = rng.choice(patient_ids, size=n, replace=True)

    age_by_patient = {pid: int(rng.integers(18, 80)) for pid in patient_ids}
    gender_by_patient = {pid: str(rng.choice(["Male", "Female"])) for pid in patient_ids}

    age = np.array([age_by_patient[pid] for pid in assigned_patients])
    gender = np.array([gender_by_patient[pid] for pid in assigned_patients])

    patient_trend = {pid: float(rng.normal(0, 0.6)) for pid in patient_ids}
    trend_signal = np.array([patient_trend[pid] for pid in assigned_patients])

    hemoglobin = rng.normal(13.2, 1.4, n) - (gender == "Female") * 0.7 + trend_signal * 0.6
    wbc = rng.normal(7.0, 2.0, n) + (trend_signal < -0.4) * 0.9
    rbc = rng.normal(4.8, 0.6, n)
    platelets = rng.normal(280, 60, n)
    cholesterol = rng.normal(190, 35, n) + (age > 45) * 15 + (trend_signal > 0.4) * 10
    hdl = rng.normal(50, 12, n)
    ldl = rng.normal(120, 30, n) + (age > 45) * 12 + (trend_signal > 0.4) * 10
    triglycerides = rng.normal(140, 45, n) + (trend_signal > 0.4) * 8

    symptoms = np.where(hemoglobin < 11.5, "Fatigue", np.where(wbc > 11, "Fever", "None"))

    days_ago = rng.integers(0, 365, n)
    test_dates = pd.Timestamp("today").normalize() - pd.to_timedelta(days_ago, unit="D")

    df = pd.DataFrame(
        {
            "Patient_ID": assigned_patients,
            "Test_Date": test_dates,
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
    ).sort_values(["Patient_ID", "Test_Date"], ignore_index=True)

    return df

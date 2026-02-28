import pandas as pd

from src.interpreter import interpret_row


def test_interpret_row_low_hemoglobin_sets_anemia_signal():
    row = pd.Series({"Hemoglobin": 10.8, "WBC": 8, "LDL": 100, "Triglycerides": 120, "Age": 28, "Gender": "Female"})
    insight = interpret_row(row)
    assert insight.anemia_risk > 0.5
    assert "hemoglobin" in insight.narrative.lower()

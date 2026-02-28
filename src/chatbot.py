from __future__ import annotations

import pandas as pd


def answer_question(question: str, latest_row: pd.Series) -> str:
    q = question.lower()
    if "wbc" in q:
        wbc = latest_row.get("WBC", "unknown")
        if float(wbc) > 11:
            return f"Your WBC is {wbc}, which is higher than typical range and can indicate infection/inflammation."
        return f"Your WBC is {wbc}, which is generally in expected range."
    if "hemoglobin" in q or "anemia" in q:
        h = latest_row.get("Hemoglobin", "unknown")
        return f"Your hemoglobin is {h}. Low values may relate to anemia, especially with fatigue symptoms."
    if "cholesterol" in q or "ldl" in q:
        l = latest_row.get("LDL", "unknown")
        c = latest_row.get("Cholesterol", "unknown")
        return f"Your LDL is {l} and total cholesterol is {c}; improving diet and exercise can lower risk."
    return "I can explain WBC, hemoglobin/anemia, and cholesterol/LDL questions based on your report."

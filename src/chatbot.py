from __future__ import annotations

import pandas as pd


def _safe_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def answer_question(question: str, latest_row: pd.Series) -> str:
    q = question.lower()

    if "wbc" in q:
        wbc_raw = latest_row.get("WBC")
        wbc = _safe_float(wbc_raw)
        if wbc is None:
            return "I couldn't read your WBC value from the latest report."
        if wbc > 11:
            return f"Your WBC is {wbc:.1f}, which is higher than typical range and can indicate infection/inflammation."
        return f"Your WBC is {wbc:.1f}, which is generally in expected range."

    if "hemoglobin" in q or "anemia" in q:
        h_raw = latest_row.get("Hemoglobin")
        h = _safe_float(h_raw)
        if h is None:
            return "I couldn't read your hemoglobin value from the latest report."
        return f"Your hemoglobin is {h:.1f}. Low values may relate to anemia, especially with fatigue symptoms."

    if "cholesterol" in q or "ldl" in q:
        l = _safe_float(latest_row.get("LDL"))
        c = _safe_float(latest_row.get("Cholesterol"))
        if l is None or c is None:
            return "I couldn't read LDL/cholesterol from the latest report."
        return f"Your LDL is {l:.0f} and total cholesterol is {c:.0f}; improving diet and exercise can lower risk."

    return "I can explain WBC, hemoglobin/anemia, and cholesterol/LDL questions based on your report."

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Insight:
    anemia_risk: float
    cardio_risk: float
    infection_risk: float
    severity_score: int
    narrative: str
    diet_tips: list[str]


def rule_based_flags(row: pd.Series) -> dict[str, bool]:
    female = str(row.get("Gender", "")).lower() == "female"
    anemia = (female and row.get("Hemoglobin", 99) < 12) or (not female and row.get("Hemoglobin", 99) < 13)
    infection = row.get("WBC", 0) > 11
    cardio = row.get("LDL", 0) > 130 or row.get("Triglycerides", 0) > 150 or row.get("Cholesterol", 0) > 200
    return {"anemia": anemia, "infection": infection, "cardio": cardio}


def severity_score(row: pd.Series) -> int:
    score = 0
    score += max(0, 12 - float(row.get("Hemoglobin", 12))) * 8
    score += max(0, float(row.get("LDL", 100)) - 100) * 0.2
    score += max(0, float(row.get("Triglycerides", 120)) - 120) * 0.15
    score += max(0, float(row.get("WBC", 7)) - 10) * 4
    if int(row.get("Age", 30)) > 50:
        score += 7
    return int(np.clip(score, 0, 100))


def lifestyle_suggestions(flags: dict[str, bool]) -> list[str]:
    tips: list[str] = []
    if flags["anemia"]:
        tips.append("Add iron-rich foods (spinach, lentils, dates) and vitamin C sources.")
    if flags["cardio"]:
        tips.append("Reduce fried foods, increase fiber, and aim for 150 min/week exercise.")
    if flags["infection"]:
        tips.append("Hydrate well, prioritize sleep, and consult a clinician if fever persists.")
    if not tips:
        tips.append("Maintain balanced diet, regular movement, and annual preventive checkups.")
    return tips


def age_band(age: int) -> str:
    lo = int(age // 10 * 10)
    return f"{lo}-{lo + 9}"


def build_narrative(row: pd.Series, flags: dict[str, bool], score: int) -> str:
    parts = []
    if flags["anemia"]:
        parts.append("Your hemoglobin is below the expected range and may indicate mild anemia.")
    else:
        parts.append("Your hemoglobin appears within expected range.")

    if flags["cardio"]:
        parts.append(f"Your lipid values are borderline high for age group {age_band(int(row.get('Age', 30)))}.")
    if flags["infection"]:
        parts.append("Your WBC is elevated, which can appear in infection or inflammation.")

    parts.append(f"Current overall severity score: {score}/100.")
    return " ".join(parts)


def interpret_row(row: pd.Series) -> Insight:
    flags = rule_based_flags(row)
    score = severity_score(row)
    narrative = build_narrative(row, flags, score)

    return Insight(
        anemia_risk=0.75 if flags["anemia"] else 0.15,
        cardio_risk=0.72 if flags["cardio"] else 0.2,
        infection_risk=0.7 if flags["infection"] else 0.18,
        severity_score=score,
        narrative=narrative,
        diet_tips=lifestyle_suggestions(flags),
    )

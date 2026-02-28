import pytest

pd = pytest.importorskip("pandas")

from src.chatbot import answer_question
from src.data_pipeline import parse_report_text
from src.interpreter import interpret_row


def test_interpret_row_low_hemoglobin_sets_anemia_signal():
    row = pd.Series({"Hemoglobin": 10.8, "WBC": 8, "LDL": 100, "Triglycerides": 120, "Age": 28, "Gender": "Female"})
    insight = interpret_row(row)
    assert insight.anemia_risk > 0.5
    assert "hemoglobin" in insight.narrative.lower()


def test_chatbot_handles_missing_numeric_values_gracefully():
    row = pd.Series({"WBC": "unknown", "Hemoglobin": None, "LDL": "n/a", "Cholesterol": "n/a"})
    assert "couldn't read" in answer_question("why is my wbc high", row).lower()
    assert "couldn't read" in answer_question("do i have anemia", row).lower()
    assert "couldn't read" in answer_question("my cholesterol", row).lower()


def test_parse_report_text_extracts_patient_and_metrics():
    text = """
    Patient ID: p-909
    Gender: female
    Age: 31
    Hemoglobin: 11.2
    WBC: 12.1
    LDL: 141
    """
    parsed = parse_report_text(text)
    assert parsed.iloc[0]["Patient_ID"] == "P-909"
    assert parsed.iloc[0]["Gender"] == "Female"
    assert float(parsed.iloc[0]["Hemoglobin"]) == 11.2


def test_synthetic_data_contains_patient_histories():
    from src.synthetic_data import generate_synthetic_dataset

    df = generate_synthetic_dataset(200, seed=7)
    counts = df["Patient_ID"].value_counts()
    assert counts.max() >= 2
    assert df["Test_Date"].is_monotonic_increasing is False

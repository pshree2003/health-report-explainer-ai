from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.chatbot import answer_question
from src.data_pipeline import StorageConfig, extract_text_from_upload, init_db, load_reports, parse_report_text, save_report
from src.interpreter import interpret_row
from src.modeling import FEATURES, shap_summary, train_models
from src.synthetic_data import generate_synthetic_dataset

st.set_page_config(page_title="AI Health Report Explainer", layout="wide")
st.title("🩺 AI-Based Health Report Explainer")

config = StorageConfig()
init_db(config)

with st.sidebar:
    st.header("Data Source")
    use_synthetic = st.checkbox("Use synthetic demo dataset", value=True)

if use_synthetic:
    synthetic_all = generate_synthetic_dataset(600)
    patient_options = sorted(synthetic_all["Patient_ID"].unique().tolist())
    selected_patient = st.sidebar.selectbox("Synthetic patient", patient_options, index=0)
    df = synthetic_all[synthetic_all["Patient_ID"] == selected_patient].copy()
else:
    upload = st.file_uploader("Upload report (txt, pdf, png, jpg)", type=["txt", "pdf", "png", "jpg", "jpeg"])
    if upload:
        raw = upload.read()
        try:
            text = extract_text_from_upload(raw, upload.name)
            parsed = parse_report_text(text)
            st.subheader("Parsed report")
            st.dataframe(parsed)

            patient_id = st.text_input("Patient ID", value=str(parsed.iloc[0].get("Patient_ID", "P-CUSTOM")))
            test_date = st.date_input("Test date")
            if st.button("Save report"):
                save_report(parsed, patient_id, test_date.isoformat(), config)
                st.success("Report encrypted and stored in SQLite")
        except Exception as exc:
            st.error(f"Could not process uploaded report: {exc}")

    patient_filter = st.text_input("Load history for Patient ID")
    df = load_reports(patient_filter or None, config)
    if df.empty:
        st.info("No uploaded reports found, switch to synthetic mode for a full demo.")

if not df.empty:
    artifacts = None
    if len(df) >= 10:
        artifacts = train_models(df)
        st.subheader("Model performance (AUC)")
        st.write(artifacts.metrics)
    else:
        st.info("Need at least 10 reports to train ML risk models; showing rule-based insights only.")

    latest = df.sort_values("Test_Date").iloc[-1]
    insight = interpret_row(latest)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Personalized explanation")
        st.info(insight.narrative)
        st.markdown("**Diet / lifestyle suggestions**")
        for tip in insight.diet_tips:
            st.write(f"- {tip}")

        summary_lines = [
            f"Patient: {latest.get('Patient_ID', 'Unknown')}",
            f"Date: {latest.get('Test_Date', 'Unknown')}",
            f"Severity Score: {insight.severity_score}/100",
            f"Anemia Risk: {insight.anemia_risk:.0%}",
            f"Cardio Risk: {insight.cardio_risk:.0%}",
            f"Infection Risk: {insight.infection_risk:.0%}",
            "",
            "Explanation:",
            insight.narrative,
            "",
            "Suggestions:",
            *[f"- {tip}" for tip in insight.diet_tips],
        ]
        st.download_button(
            "Download patient-friendly summary",
            data="\n".join(summary_lines),
            file_name="health_summary.txt",
            mime="text/plain",
        )

    with col2:
        st.metric("Severity Score", f"{insight.severity_score}/100")
        st.metric("Anemia risk", f"{insight.anemia_risk:.0%}")
        st.metric("Cardio risk", f"{insight.cardio_risk:.0%}")
        st.metric("Infection risk", f"{insight.infection_risk:.0%}")

    st.subheader("Health trend analyzer")
    trend_cols = [c for c in ["Hemoglobin", "WBC", "LDL", "Cholesterol", "Triglycerides"] if c in df.columns]
    long_df = df[["Test_Date"] + trend_cols].copy()
    long_df["Test_Date"] = pd.to_datetime(long_df["Test_Date"])
    melted = long_df.melt(id_vars="Test_Date", var_name="Marker", value_name="Value")
    fig = px.line(melted, x="Test_Date", y="Value", color="Marker", markers=True)
    st.plotly_chart(fig, use_container_width=True)

    if len(df) >= 3:
        hb = df.sort_values("Test_Date")["Hemoglobin"].tail(3).tolist()
        if hb[0] > hb[1] > hb[2]:
            st.warning("Early warning: Hemoglobin declined across the last 3 reports. Consider medical review.")

    st.subheader("Explainable AI")
    if artifacts is not None:
        st.caption(shap_summary(artifacts.cardio_model, df[FEATURES].head(80).fillna(df[FEATURES].median(numeric_only=True))))
    else:
        st.caption("SHAP summary requires a trained model (add more reports).")

    st.subheader("Medical NLP chatbot")
    question = st.text_input("Ask: Why is my WBC high?")
    if question:
        st.success(answer_question(question, latest))

st.caption("Privacy note: reports are encrypted at rest in local SQLite (HIPAA-style design simulation).")

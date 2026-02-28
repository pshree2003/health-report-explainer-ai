# AI-Based Health Report Explainer

Advanced student-ready healthcare AI project that:

- Parses reports from text/PDF/image via OCR + regex.
- Interprets medical values using a hybrid rule + ML approach.
- Generates personalized plain-English insights with severity score.
- Tracks time trends and raises simple early warnings.
- Trains risk models (anemia, cardiovascular, infection).
- Provides Explainable AI notes via SHAP.
- Includes a Streamlit dashboard, SQLite storage, and local encryption.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Data

By default, the app uses synthetic data generated from realistic ranges.
You can switch to custom report upload mode from the sidebar.

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

try:
    import shap
except Exception:  # pragma: no cover
    shap = None


FEATURES = [
    "Hemoglobin",
    "WBC",
    "RBC",
    "Platelets",
    "Cholesterol",
    "HDL",
    "LDL",
    "Triglycerides",
    "Age",
]


@dataclass
class ModelArtifacts:
    anemia_model: Pipeline
    cardio_model: Pipeline
    infection_model: DecisionTreeClassifier
    metrics: dict[str, float]


def train_models(df: pd.DataFrame) -> ModelArtifacts:
    x = df[FEATURES].fillna(df[FEATURES].median())

    y_anemia = (df["Hemoglobin"] < 12).astype(int)
    y_cardio = ((df["LDL"] > 130) | (df["Cholesterol"] > 200) | (df["Triglycerides"] > 150)).astype(int)
    y_infection = (df["WBC"] > 11).astype(int)

    xa_train, xa_test, ya_train, ya_test = train_test_split(x, y_anemia, test_size=0.2, random_state=42)
    xc_train, xc_test, yc_train, yc_test = train_test_split(x, y_cardio, test_size=0.2, random_state=42)
    xi_train, xi_test, yi_train, yi_test = train_test_split(x, y_infection, test_size=0.2, random_state=42)

    logistic = lambda: Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=400))])

    anemia_model = logistic().fit(xa_train, ya_train)
    cardio_model = logistic().fit(xc_train, yc_train)
    infection_model = DecisionTreeClassifier(max_depth=4, random_state=42).fit(xi_train, yi_train)

    metrics = {
        "anemia_auc": roc_auc_score(ya_test, anemia_model.predict_proba(xa_test)[:, 1]),
        "cardio_auc": roc_auc_score(yc_test, cardio_model.predict_proba(xc_test)[:, 1]),
        "infection_auc": roc_auc_score(yi_test, infection_model.predict_proba(xi_test)[:, 1]),
    }

    return ModelArtifacts(anemia_model=anemia_model, cardio_model=cardio_model, infection_model=infection_model, metrics=metrics)


def shap_summary(model: Pipeline, x_sample: pd.DataFrame) -> str:
    if shap is None:
        return "SHAP not available in current environment."
    estimator = model.named_steps["clf"]
    explainer = shap.Explainer(estimator, x_sample)
    vals = explainer(x_sample)
    impact = abs(vals.values).mean(axis=0)
    top_idx = impact.argmax()
    pct = impact[top_idx] / (impact.sum() + 1e-9) * 100
    return f"Top risk influence: {x_sample.columns[top_idx]} contributes approximately {pct:.1f}% of model signal."

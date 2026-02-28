from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.dummy import DummyClassifier
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
    infection_model: DecisionTreeClassifier | DummyClassifier
    metrics: dict[str, float]


def _safe_auc(y_true: pd.Series, probas: pd.Series) -> float:
    if y_true.nunique() < 2:
        return 0.5
    return float(roc_auc_score(y_true, probas))


def _fit_logistic_or_dummy(x_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    if y_train.nunique() < 2:
        return Pipeline([("clf", DummyClassifier(strategy="most_frequent"))]).fit(x_train, y_train)
    return Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=400))]).fit(x_train, y_train)


def _predict_positive_proba(model: Pipeline | DecisionTreeClassifier | DummyClassifier, x_test: pd.DataFrame) -> pd.Series:
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(x_test)
        if probs.shape[1] == 1:
            return pd.Series([0.0] * len(x_test), index=x_test.index)
        return pd.Series(probs[:, 1], index=x_test.index)
    return pd.Series([0.0] * len(x_test), index=x_test.index)


def train_models(df: pd.DataFrame) -> ModelArtifacts:
    x = df[FEATURES].fillna(df[FEATURES].median(numeric_only=True))

    y_anemia = (df["Hemoglobin"] < 12).astype(int)
    y_cardio = ((df["LDL"] > 130) | (df["Cholesterol"] > 200) | (df["Triglycerides"] > 150)).astype(int)
    y_infection = (df["WBC"] > 11).astype(int)

    xa_train, xa_test, ya_train, ya_test = train_test_split(x, y_anemia, test_size=0.2, random_state=42)
    xc_train, xc_test, yc_train, yc_test = train_test_split(x, y_cardio, test_size=0.2, random_state=42)
    xi_train, xi_test, yi_train, yi_test = train_test_split(x, y_infection, test_size=0.2, random_state=42)

    anemia_model = _fit_logistic_or_dummy(xa_train, ya_train)
    cardio_model = _fit_logistic_or_dummy(xc_train, yc_train)

    if yi_train.nunique() < 2:
        infection_model: DecisionTreeClassifier | DummyClassifier = DummyClassifier(strategy="most_frequent").fit(xi_train, yi_train)
    else:
        infection_model = DecisionTreeClassifier(max_depth=4, random_state=42).fit(xi_train, yi_train)

    metrics = {
        "anemia_auc": _safe_auc(ya_test, _predict_positive_proba(anemia_model, xa_test)),
        "cardio_auc": _safe_auc(yc_test, _predict_positive_proba(cardio_model, xc_test)),
        "infection_auc": _safe_auc(yi_test, _predict_positive_proba(infection_model, xi_test)),
    }

    return ModelArtifacts(anemia_model=anemia_model, cardio_model=cardio_model, infection_model=infection_model, metrics=metrics)


def shap_summary(model: Pipeline, x_sample: pd.DataFrame) -> str:
    if shap is None:
        return "SHAP not available in current environment."

    if not isinstance(model, Pipeline) or "clf" not in model.named_steps:
        return "SHAP summary unavailable for non-pipeline model."

    estimator = model.named_steps["clf"]
    if "scale" in model.named_steps:
        x_for_shap = pd.DataFrame(model.named_steps["scale"].transform(x_sample), columns=x_sample.columns)
    else:
        x_for_shap = x_sample

    explainer = shap.Explainer(estimator, x_for_shap)
    vals = explainer(x_for_shap)
    impact = abs(vals.values).mean(axis=0)
    top_idx = int(impact.argmax())
    pct = float(impact[top_idx] / (impact.sum() + 1e-9) * 100)
    return f"Top risk influence: {x_sample.columns[top_idx]} contributes approximately {pct:.1f}% of model signal."

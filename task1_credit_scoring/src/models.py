"""
Model definitions for credit scoring.

Each model is wrapped in a sklearn Pipeline (StandardScaler + Classifier)
so that raw (unscaled) data can be passed directly.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from typing import Dict


MODEL_CONFIGS = {
    "Logistic Regression": LogisticRegression(
        max_iter=500, random_state=42, solver="lbfgs"
    ),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=8, random_state=42, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42
    ),
}


def build_models() -> Dict[str, Pipeline]:
    """Return a dict of named sklearn Pipelines.

    Each pipeline applies StandardScaler followed by the classifier,
    so the caller can pass raw (unscaled) features.

    Returns
    -------
    dict[str, Pipeline]
        Mapping of model name → Pipeline.
    """
    pipelines: Dict[str, Pipeline] = {}
    for name, clf in MODEL_CONFIGS.items():
        pipelines[name] = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", clf),
            ]
        )
    return pipelines


def train_models(
    pipelines: Dict[str, Pipeline],
    X_train,
    y_train,
) -> Dict[str, Pipeline]:
    """Fit every pipeline on the training data.

    Parameters
    ----------
    pipelines : dict[str, Pipeline]
        Output of ``build_models()``.
    X_train : array-like
        Training features (raw / unscaled is fine because pipelines scale internally).
    y_train : array-like
        Training labels.

    Returns
    -------
    dict[str, Pipeline]
        The same dict, now with fitted pipelines.
    """
    for name, pipe in pipelines.items():
        print(f"  Training {name} …")
        pipe.fit(X_train, y_train)
    return pipelines


if __name__ == "__main__":
    models = build_models()
    for name, pipe in models.items():
        print(f"{name}: {pipe}")

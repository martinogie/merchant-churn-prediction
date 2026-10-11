"""Random Forest churn model trained on SMOTE-balanced data with a time-based split."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)

from .features import encode
from .smote import smote


class ChurnModel:
    def __init__(self, smote_ratio: float = 0.5, seed: int = 11, **rf_params):
        self.smote_ratio, self.seed = smote_ratio, seed
        self.rf_params = dict(n_estimators=300, max_depth=12, min_samples_leaf=20,
                              max_features="sqrt", n_jobs=-1, random_state=seed)
        self.rf_params.update(rf_params)
        self.threshold = 0.5

    def fit(self, train: pd.DataFrame, valid: pd.DataFrame) -> "ChurnModel":
        X = encode(train)
        self.columns = list(X.columns)
        Xs, ys = smote(X.to_numpy(), train["churn"].to_numpy(), ratio=self.smote_ratio, seed=self.seed)
        self.rf = RandomForestClassifier(**self.rf_params).fit(Xs, ys)
        # Pick the alert threshold that maximises F1 on a later validation month (never the test set).
        p = self.predict_proba(valid)
        grid = np.linspace(0.05, 0.95, 91)
        f1s = [f1_score(valid["churn"], p >= t, zero_division=0) for t in grid]
        self.threshold = float(grid[int(np.argmax(f1s))])
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.rf.predict_proba(encode(df, self.columns).to_numpy())[:, 1]

    def evaluate(self, df: pd.DataFrame) -> dict:
        y, p = df["churn"].to_numpy(), self.predict_proba(df)
        yhat = p >= self.threshold
        return {"accuracy": accuracy_score(y, yhat), "precision": precision_score(y, yhat, zero_division=0),
                "recall": recall_score(y, yhat), "f1": f1_score(y, yhat), "roc_auc": roc_auc_score(y, p),
                "pr_auc": average_precision_score(y, p), "threshold": self.threshold,
                "base_rate": float(y.mean()), "n": int(len(y))}

    def importance(self) -> pd.Series:
        return pd.Series(self.rf.feature_importances_, index=self.columns).sort_values(ascending=False)

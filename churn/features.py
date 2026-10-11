"""Turn the monthly panel into model-ready snapshots.

A snapshot is one merchant at the end of month t, using only months <= t. The label is whether the
merchant has churned (zero volume, which is permanent in this panel) by month t + 3.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import SEASONALITY

NUMERIC = ["log_tpv", "tpv_trend_sa", "tpv_vs_3m", "txn_count", "active_days", "active_days_change",
           "failed_rate_3m", "payout_delay_3m", "tickets_3m", "chargebacks_3m", "tenure_months"]
CATEGORICAL = ["segment", "integration", "plan"]
HORIZON = 3


def build_snapshots(panel: pd.DataFrame, months: range) -> pd.DataFrame:
    p = panel.sort_values(["merchant_id", "month"]).copy()
    p["seas"] = SEASONALITY[p["month"] - 1]
    p["log_tpv"] = np.log1p(p["tpv"])
    p["log_tpv_sa"] = np.log1p(p["tpv"] / p["seas"])
    g = p.groupby("merchant_id")
    for col, src in [("failed_rate_3m", "failed_rate"), ("payout_delay_3m", "payout_delay_days"),
                     ("tickets_3m", "support_tickets"), ("chargebacks_3m", "chargebacks")]:
        p[col] = g[src].transform(lambda s: s.rolling(3, min_periods=1).mean())
    p["tpv_3m_prior"] = g["log_tpv"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    p["tpv_vs_3m"] = p["log_tpv"] - p["tpv_3m_prior"]
    p["tpv_trend_sa"] = p["log_tpv_sa"] - g["log_tpv_sa"].shift(1)        # seasonally adjusted change
    p["active_days_change"] = p["active_days"] - g["active_days"].shift(1)
    p["tenure_months"] = p["tenure_start_months"] + p["month"]
    p["churn"] = (g["tpv"].shift(-HORIZON) == 0).astype(int)

    snap = p[p["month"].isin(months) & (p["tpv"] > 0)].dropna(subset=["tpv_vs_3m", "tpv_trend_sa"])
    return snap[["merchant_id", "month", *NUMERIC, *CATEGORICAL, "tpv", "churn"]].reset_index(drop=True)


def encode(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    X = pd.get_dummies(df[NUMERIC + CATEGORICAL], columns=CATEGORICAL, dtype=float)
    if columns is not None:
        X = X.reindex(columns=columns, fill_value=0.0)
    return X

"""Separate seasonal decline from product friction with difference-in-differences.

Outcome: log(1 + monthly TPV) for merchant i in month t.
Treated: plugin-integration merchants, exposed to the new payout-verification flow from July 2025.
Two-way fixed effects (merchant + month) absorb each merchant's level and every month's seasonality,
so the treated x post coefficient isolates the friction effect. Standard errors are clustered by merchant.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .data import FRICTION_MONTH


def _twfe_demean(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df[cols].copy()
    for c in cols:  # balanced panel: one pass of two-way demeaning is exact
        out[c] = (df[c] - df.groupby("merchant_id")[c].transform("mean")
                  - df.groupby("month")[c].transform("mean") + df[c].mean())
    return out


def _ols_cluster(y: np.ndarray, X: np.ndarray, clusters: np.ndarray, dof_extra: int = 0):
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    meat = np.zeros((X.shape[1], X.shape[1]))
    order = np.argsort(clusters, kind="stable")
    Xo, ro, co = X[order], resid[order], clusters[order]
    bounds = np.flatnonzero(np.diff(co)) + 1
    for Xg, rg in zip(np.split(Xo, bounds), np.split(ro, bounds)):
        s = Xg.T @ rg
        meat += np.outer(s, s)
    G, n, k = len(bounds) + 1, len(y), X.shape[1] + dof_extra
    vcov = XtX_inv @ meat @ XtX_inv * (G / (G - 1)) * ((n - 1) / (n - k))
    return beta, np.sqrt(np.diag(vcov)), G


def diff_in_diff(panel: pd.DataFrame, survivors_only: bool = True) -> dict:
    """Average treatment effect of the friction event on log TPV.

    survivors_only=True restricts to merchants active all year, so the estimate measures the effect on
    volume of merchants who stayed (churn is analysed separately by the model)."""
    p = panel.copy()
    if survivors_only:
        keep = p.groupby("merchant_id")["tpv"].transform("min") > 0
        p = p[keep]
    p["y"] = np.log1p(p["tpv"])
    p["did"] = ((p["integration"] == "plugin") & (p["month"] >= FRICTION_MONTH)).astype(float)
    d = _twfe_demean(p, ["y", "did"])
    n_m, n_t = p["merchant_id"].nunique(), p["month"].nunique()
    beta, se, G = _ols_cluster(d["y"].to_numpy(), d[["did"]].to_numpy(), p["merchant_id"].to_numpy(),
                               dof_extra=n_m + n_t - 1)
    t = beta[0] / se[0]
    return {"effect_log_tpv": float(beta[0]), "se": float(se[0]), "t": float(t),
            "p_value": float(2 * stats.t.sf(abs(t), G - 1)),
            "ci95": [float(beta[0] - 1.96 * se[0]), float(beta[0] + 1.96 * se[0])],
            "effect_pct_tpv": float(np.expm1(beta[0])), "merchants": int(n_m)}


def event_study(panel: pd.DataFrame) -> pd.DataFrame:
    """Treated-minus-control gap by month, relative to the average pre-event gap (Jan-Jun)."""
    p = panel[panel.groupby("merchant_id")["tpv"].transform("min") > 0].copy()
    p["y"] = np.log1p(p["tpv"])
    p["treated"] = p["integration"] == "plugin"
    gap = p.groupby(["month", "treated"])["y"].mean().unstack()
    gap = (gap[True] - gap[False])
    gap = gap - gap.loc[: FRICTION_MONTH - 1].mean()
    return gap.rename("treated_minus_control").reset_index()


def pretrend_test(panel: pd.DataFrame) -> dict:
    """Do treated and control merchants trend the same way before the event? (slope difference = 0)."""
    p = panel[(panel["month"] < FRICTION_MONTH)]
    p = p[p.groupby("merchant_id")["tpv"].transform("min") > 0].copy()
    p["y"] = np.log1p(p["tpv"])
    slopes = p.groupby("merchant_id").apply(lambda g: np.polyfit(g["month"], g["y"], 1)[0], include_groups=False)
    plugin = panel.drop_duplicates("merchant_id").set_index("merchant_id")["integration"].eq("plugin")
    a, b = slopes[plugin.reindex(slopes.index)], slopes[~plugin.reindex(slopes.index)]
    t, pv = stats.ttest_ind(a, b, equal_var=False)
    return {"slope_treated": float(a.mean()), "slope_control": float(b.mean()), "t": float(t), "p_value": float(pv)}


def naive_before_after(panel: pd.DataFrame) -> dict:
    """What a simple before/after comparison of plugin merchants would (wrongly) conclude."""
    p = panel[(panel["integration"] == "plugin") & (panel["tpv"] > 0)].copy()
    p["y"] = np.log1p(p["tpv"])
    pre = p[p["month"].between(FRICTION_MONTH - 3, FRICTION_MONTH - 1)]["y"].mean()
    post = p[p["month"].between(FRICTION_MONTH, FRICTION_MONTH + 2)]["y"].mean()
    return {"naive_change_log_tpv": float(post - pre)}

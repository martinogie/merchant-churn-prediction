"""Randomised test of the churn early-warning workflow (account-manager outreach).

High-risk merchants flagged by the model are randomly split: treatment gets proactive outreach within
24 hours, control gets business as usual. Outcomes here are simulated from a known treatment effect so
the analysis code can be verified; with real data, replace `simulate_outcomes` with observed churn.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

TAKE_RATE = 0.015  # revenue = 1.5% of processed volume


def assign(flagged: pd.DataFrame, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = flagged.copy()
    out["group"] = np.where(rng.random(len(out)) < 0.5, "treatment", "control")
    return out


def simulate_outcomes(assigned: pd.DataFrame, relative_reduction: float = 0.25, seed: int = 4) -> pd.DataFrame:
    """Control keeps its observed outcome; in treatment, each merchant who would have churned is retained
    with probability `relative_reduction` (the assumed effect of outreach)."""
    rng = np.random.default_rng(seed)
    out = assigned.copy()
    saved = (out["group"] == "treatment") & (out["churn"] == 1) & (rng.random(len(out)) < relative_reduction)
    out["churned"] = np.where(saved, 0, out["churn"]).astype(int)
    return out


def analyse(results: pd.DataFrame) -> dict:
    t = results[results["group"] == "treatment"]
    c = results[results["group"] == "control"]
    p1, p0, n1, n0 = t["churned"].mean(), c["churned"].mean(), len(t), len(c)
    pooled = results["churned"].mean()
    se_pooled = np.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n0))
    z = (p1 - p0) / se_pooled
    se = np.sqrt(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0)
    diff = p1 - p0
    annual_rev = (results["tpv"] * 12 * TAKE_RATE).mean()
    saved = -diff * n1
    return {"n_treatment": int(n1), "n_control": int(n0), "churn_treatment": float(p1), "churn_control": float(p0),
            "absolute_reduction": float(-diff), "relative_reduction": float(-diff / p0) if p0 else 0.0,
            "z": float(z), "p_value": float(2 * stats.norm.sf(abs(z))),
            "ci95_difference": [float(diff - 1.96 * se), float(diff + 1.96 * se)],
            "merchants_retained": float(saved),
            "annual_revenue_retained_usd": float(saved * annual_rev)}

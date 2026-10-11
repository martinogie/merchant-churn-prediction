"""Synthetic merchant activity panel for a payments company.

10,000 merchants observed monthly for 12 months (Jan-Dec 2025). The generator builds in:
  * seasonality (strong Nov-Dec, weak Jan-Feb), which by itself looks like "decline";
  * a product-friction event: from July, merchants on the plugin integration go through a new
    payout-verification flow that raises failed payments and payout delays;
  * a churn process driven by falling volume, failures, payout delays and support contacts.
Because the true effects are known, the causal and predictive analyses can be checked against them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MONTHS = 12
FRICTION_MONTH = 7          # July 2025: new payout verification for plugin merchants
SEASONALITY = np.array([0.86, 0.88, 0.97, 0.99, 1.00, 0.98, 0.97, 0.99, 1.01, 1.06, 1.15, 1.24])
FRICTION_TPV_EFFECT = -0.08  # true causal effect on log TPV for treated merchants after the event


def generate_panel(n_merchants: int = 10_000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    m = n_merchants
    segment = rng.choice(["SMB", "Mid-market", "Enterprise"], m, p=[0.72, 0.22, 0.06])
    integration = rng.choice(["plugin", "api", "pos"], m, p=[0.45, 0.40, 0.15])
    plan = rng.choice(["standard", "growth", "custom"], m, p=[0.60, 0.30, 0.10])
    industry = rng.choice(["retail", "food", "services", "education", "travel"], m)
    tenure0 = rng.integers(1, 60, m)
    base = {"SMB": 9.6, "Mid-market": 11.6, "Enterprise": 13.4}
    log_tpv0 = np.array([base[s] for s in segment]) + rng.normal(0, 0.7, m)
    merchant_trend = rng.normal(0.0, 0.025, m)          # some merchants grow, some fade
    frailty = rng.normal(0, 0.6, m)                      # unobserved churn propensity

    rows, alive = [], np.ones(m, bool)
    # Merchants rarely vanish overnight: once a merchant decides to leave, volume fades for 1-3 months.
    leaving_in = np.full(m, -1)          # months until churn (-1 = not leaving)
    fade = np.zeros(m)                   # cumulative log-volume drop while leaving
    treated = integration == "plugin"
    prev_log_tpv = log_tpv0.copy()
    for t in range(1, MONTHS + 1):
        post = t >= FRICTION_MONTH
        friction = treated & post
        failed_rate = np.clip(rng.beta(2, 60, m) + 0.025 * friction, 0, 0.5)
        payout_delay = np.clip(rng.gamma(1.5, 0.6, m) + 1.4 * friction, 0, None)
        tickets = rng.poisson(0.25 + 4.0 * failed_rate + 0.15 * payout_delay + 0.4 * (leaving_in > 0))
        chargebacks = rng.poisson(np.exp(prev_log_tpv - 12.5) * 0.8)
        leaving = leaving_in > 0
        fade = np.where(leaving, fade + rng.uniform(0.25, 0.6, m), fade)
        log_tpv = (log_tpv0 + merchant_trend * t + np.log(SEASONALITY[t - 1])
                   + FRICTION_TPV_EFFECT * friction - fade + rng.normal(0, 0.18, m))
        tpv = np.where(alive, np.exp(log_tpv), 0.0)
        txn = np.where(alive, rng.poisson(np.maximum(tpv / rng.uniform(35, 90, m), 1)), 0)
        p_day = np.clip(0.35 + 0.05 * (log_tpv - 9) - 0.12 * (leaving_in > 0), 0.05, 0.98)
        active_days = np.where(alive, np.clip(rng.binomial(30, p_day), 1, 30), 0)

        rows.append(pd.DataFrame({
            "merchant_id": np.arange(m), "month": t, "tpv": tpv.round(2), "txn_count": txn,
            "active_days": active_days, "failed_rate": np.where(alive, failed_rate, 0).round(4),
            "payout_delay_days": np.where(alive, payout_delay, 0).round(2),
            "support_tickets": np.where(alive, tickets, 0), "chargebacks": np.where(alive, chargebacks, 0),
        }))

        # Monthly churn hazard for merchants still active.
        trend = log_tpv - prev_log_tpv - np.log(SEASONALITY[t - 1] / SEASONALITY[t - 2] if t > 1 else 1)
        z = (-5.1 + frailty - 3.0 * trend + 14 * failed_rate + 0.30 * payout_delay
             + 0.20 * np.minimum(tickets, 5) - 0.012 * np.minimum(tenure0 + t, 60)
             + 0.6 * (segment == "SMB") - 0.5 * (segment == "Enterprise") - 0.3 * (plan == "custom"))
        start = alive & (leaving_in < 0) & (rng.random(m) < 1 / (1 + np.exp(-z)))
        leaving_in = np.where(start, rng.integers(1, 4, m), leaving_in)   # 1-3 months of fading
        churn_now = alive & (leaving_in == 0)
        leaving_in = np.where(leaving_in > 0, leaving_in - 1, leaving_in)
        alive &= ~churn_now
        prev_log_tpv = log_tpv

    panel = pd.concat(rows, ignore_index=True)
    static = pd.DataFrame({"merchant_id": np.arange(m), "segment": segment, "integration": integration,
                           "plan": plan, "industry": industry, "tenure_start_months": tenure0})
    return panel.merge(static, on="merchant_id")

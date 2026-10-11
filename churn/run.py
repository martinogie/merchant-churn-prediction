"""End-to-end pipeline: data -> features -> model -> causal analysis -> A/B test -> reports.

    python -m churn.run
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from . import ab_test, causal
from .data import FRICTION_MONTH, FRICTION_TPV_EFFECT, generate_panel
from .features import build_snapshots
from .model import ChurnModel

OUT = Path("reports")


def main(seed: int = 7):
    OUT.mkdir(exist_ok=True)
    panel = generate_panel(seed=seed)
    snaps = build_snapshots(panel, range(3, 10))
    train, valid, test = snaps[snaps.month <= 6], snaps[snaps.month == 7], snaps[snaps.month.isin([8, 9])]

    model = ChurnModel().fit(train, valid)
    metrics = model.evaluate(test)

    did = causal.diff_in_diff(panel)
    pre = causal.pretrend_test(panel)
    naive = causal.naive_before_after(panel)
    es = causal.event_study(panel)

    # A/B test on merchants flagged in the latest month (month 9).
    latest = test[test.month == 9].copy()
    latest["risk"] = model.predict_proba(latest)
    flagged = latest[latest["risk"] >= model.threshold]
    ab = ab_test.analyse(ab_test.simulate_outcomes(ab_test.assign(flagged)))

    results = {
        "data": {"merchants": int(panel.merchant_id.nunique()), "months": 12,
                 "train_snapshots": len(train), "valid_snapshots": len(valid), "test_snapshots": len(test),
                 "annual_churn_rate": float(1 - (panel[panel.month == 12].tpv > 0).mean())},
        "model_test": metrics,
        "top_features": model.importance().head(10).round(4).to_dict(),
        "diff_in_diff": did, "true_friction_effect_log_tpv": FRICTION_TPV_EFFECT,
        "pretrend_test": pre, "naive_before_after": naive,
        "ab_test": ab,
    }
    (OUT / "metrics.json").write_text(json.dumps(results, indent=2))
    es.to_csv(OUT / "event_study.csv", index=False)
    joblib.dump(model, OUT / "model.joblib")
    _plots(model, test, es)
    _report(results)
    m = metrics
    print(f"Model   acc {m['accuracy']:.3f}  F1 {m['f1']:.3f}  prec {m['precision']:.3f}  recall {m['recall']:.3f}  "
          f"ROC-AUC {m['roc_auc']:.3f}  PR-AUC {m['pr_auc']:.3f}  (base rate {m['base_rate']:.1%})")
    print(f"DiD     friction effect on TPV {did['effect_pct_tpv']:+.1%} (p={did['p_value']:.2g}); naive before/after "
          f"{np.expm1(naive['naive_change_log_tpv']):+.1%}; pre-trend p={pre['p_value']:.2f}")
    print(f"A/B     churn {ab['churn_control']:.1%} -> {ab['churn_treatment']:.1%} (p={ab['p_value']:.3f}), "
          f"~${ab['annual_revenue_retained_usd']:,.0f} annual revenue retained")


def _plots(model, test, es):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    y, p = test["churn"], model.predict_proba(test)
    prec, rec, _ = precision_recall_curve(y, p)
    ax[0].plot(rec, prec, color="#2F5D8A")
    ax[0].axhline(y.mean(), ls="--", color="grey", lw=1, label=f"Base rate {y.mean():.1%}")
    ax[0].set(title="Precision–recall (test months)", xlabel="Recall", ylabel="Precision")
    ax[0].legend()
    imp = model.importance().head(10)[::-1]
    ax[1].barh(imp.index, imp.values, color="#2F5D8A")
    ax[1].set(title="Top features (impurity importance)")
    ax[2].plot(es["month"], es["treated_minus_control"], "o-", color="#C2410C")
    ax[2].axvline(FRICTION_MONTH - 0.5, ls="--", color="grey", lw=1)
    ax[2].axhline(0, color="grey", lw=0.8)
    ax[2].set(title="Event study: plugin vs. other merchants", xlabel="Month (2025)",
              ylabel="Gap in log TPV vs. Jan–Jun average")
    fig.tight_layout()
    fig.savefig(OUT / "results.png", dpi=130)
    plt.close(fig)


def _report(r):
    m, d, a = r["model_test"], r["diff_in_diff"], r["ab_test"]
    lines = [
        "# Merchant Churn – Results",
        "",
        f"{r['data']['merchants']:,} merchants × 12 months · annual churn {r['data']['annual_churn_rate']:.1%} · "
        f"label: churned within 3 months · test months Aug–Sep 2025.",
        "",
        "## Model (Random Forest + SMOTE)",
        "",
        "| Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Base rate |",
        "|---|---|---|---|---|---|---|",
        f"| {m['accuracy']:.3f} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m['roc_auc']:.3f} | "
        f"{m['pr_auc']:.3f} | {m['base_rate']:.1%} |",
        "",
        "## Seasonality vs. product friction (difference-in-differences)",
        "",
        f"- Friction effect on volume: **{d['effect_pct_tpv']:+.1%}** (95% CI {np.expm1(d['ci95'][0]):+.1%} to "
        f"{np.expm1(d['ci95'][1]):+.1%}, p = {d['p_value']:.2g}); true simulated effect {np.expm1(r['true_friction_effect_log_tpv']):+.1%}.",
        f"- A naive before/after comparison shows {np.expm1(r['naive_before_after']['naive_change_log_tpv']):+.1%}, "
        "because seasonality and churn are mixed in.",
        f"- Pre-trend check: p = {r['pretrend_test']['p_value']:.2f} (no evidence of different trends before July).",
        "",
        "## A/B test of the early-warning workflow (simulated outcomes)",
        "",
        f"- Churn {a['churn_control']:.1%} (control, n={a['n_control']}) vs. {a['churn_treatment']:.1%} "
        f"(treatment, n={a['n_treatment']}); z = {a['z']:.2f}, p = {a['p_value']:.3f}.",
        f"- About {a['merchants_retained']:.0f} merchants and ${a['annual_revenue_retained_usd']:,.0f} in annual revenue retained.",
        "",
        "![results](results.png)",
    ]
    (OUT / "RESULTS.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()

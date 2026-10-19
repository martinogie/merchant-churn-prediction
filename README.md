# Merchant Churn Prediction & Real-Time Scoring

Predicts which payment merchants will stop processing within the next 3 months, separates seasonal
slowdowns from product-driven churn with **difference-in-differences**, tests the retention workflow
with a **randomised A/B test**, and serves scores through a **Flask API** (p95 < 50 ms).

**[Project page](https://martinogie.github.io/merchant-churn-prediction/)**

![Results](reports/results.png)

## Results

**Model** (Random Forest + SMOTE, trained Mar–Jun, threshold tuned on Jul, tested on Aug–Sep 2025)

| Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Churn base rate |
|---|---|---|---|---|---|---|
| 0.923 | 0.666 | 0.368 | 0.475 | 0.800 | 0.504 | 9.5% |

Accuracy looks high only because most merchants don't churn; PR-AUC (0.50 vs. a
9.5% base rate) and precision at the alert threshold are the numbers that matter for
an account-management team: about 67% of alerts are merchants who really leave.

**Seasonality vs. product friction.** In July 2025 plugin-integration merchants moved to a new payout
verification flow. Two-way fixed-effects DiD (merchant + month effects, merchant-clustered errors)
estimates the friction cut their volume by **-8.3%**
(95% CI -9.1% to -7.5%), against a true simulated effect of
-7.7%. A naive before/after comparison gives
-6.9% because it mixes in seasonality, merchant growth and merchants
fading out before churn. Pre-period trends match (p = 0.16).

**A/B test of the early-warning workflow.** Merchants flagged in September were randomised to
account-manager outreach within 24 hours or business as usual. Churn fell from
70.1% to 60.4% (z = -2.08, p = 0.038), about
20 merchants and **$53,506** in annual revenue
retained (1.5% take rate). Outcomes are simulated from an assumed effect so the analysis can be verified;
the same code runs on real experiment data.

## Project layout

| Module | What it does |
|---|---|
| `churn/data.py` | Synthetic panel: 10,000 merchants × 12 months with seasonality, a product-friction event and a churn process where volume fades 1–3 months before a merchant leaves |
| `churn/features.py` | Point-in-time snapshots (no look-ahead): volume vs. 3-month average, seasonally adjusted trend, active days, failures, payout delays, tickets |
| `churn/smote.py` | SMOTE implemented from scratch (k-NN interpolation), applied to training data only |
| `churn/model.py` | Random Forest, F1-optimal alert threshold chosen on a later validation month |
| `churn/causal.py` | Two-way FE difference-in-differences, event study, pre-trend test |
| `churn/ab_test.py` | Randomisation, two-proportion z-test, confidence interval, revenue impact |
| `churn/api.py` | Flask `/score` endpoint returning probability, risk tier and recommended action |
| `tests/` | 7 tests: permanence of churn, no leakage, SMOTE, model lift, DiD recovers the true effect, A/B power, API latency |

## Quick start

```bash
pip install -r requirements.txt
python -m churn.run                      # data, model, DiD, A/B test -> reports/
python -m unittest discover -s tests
python -m churn.api                      # http://127.0.0.1:8000
curl -X POST localhost:8000/score -H "Content-Type: application/json" -d @examples/merchant.json
```

```json
{"results": [{"merchant_id": 1042, "churn_probability": 0.82, "risk_tier": "high",
  "alert_account_manager": true, "action": "Account manager call within 24 hours"}], "latency_ms": 22.4}
```

Docker:

```bash
docker build -t merchant-churn . && docker run -p 8000:8000 merchant-churn
```

## Data note

All data is synthetic. It is generated with known effects (seasonality, a −7.7% friction effect on
volume, and fading activity before churn), so each method can be checked against the truth.

## How this was built

Built with AI-assisted coding. Modeling choices, validation and review are my own.

## License

MIT

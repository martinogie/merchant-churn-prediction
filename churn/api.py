"""Real-time churn scoring API (Flask).

    python -m churn.api            # serves on http://127.0.0.1:8000
    curl -X POST localhost:8000/score -H "Content-Type: application/json" -d @examples/merchant.json
"""
from __future__ import annotations

import os
import time

import joblib
import pandas as pd
from flask import Flask, jsonify, request

from .features import CATEGORICAL, NUMERIC

MODEL_PATH = os.environ.get("CHURN_MODEL", "reports/model.joblib")
REQUIRED = NUMERIC + CATEGORICAL


def create_app(model=None) -> Flask:
    app = Flask(__name__)
    app.config["model"] = model if model is not None else joblib.load(MODEL_PATH)
    app.config["model"].rf.n_jobs = 1   # single-request scoring is faster without thread fan-out

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/score")
    def score():
        t0 = time.perf_counter()
        payload = request.get_json(silent=True) or {}
        merchants = payload if isinstance(payload, list) else [payload]
        missing = sorted({f for m in merchants for f in REQUIRED if f not in m})
        if missing:
            return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400
        model = app.config["model"]
        probs = model.predict_proba(pd.DataFrame(merchants))
        out = []
        for m, p in zip(merchants, probs):
            tier = "high" if p >= model.threshold else ("medium" if p >= model.threshold / 2 else "low")
            out.append({"merchant_id": m.get("merchant_id"), "churn_probability": round(float(p), 4),
                        "risk_tier": tier, "alert_account_manager": bool(p >= model.threshold),
                        "action": {"high": "Account manager call within 24 hours",
                                   "medium": "Add to weekly check-in list",
                                   "low": "No action"}[tier]})
        ms = (time.perf_counter() - t0) * 1000
        return jsonify({"results": out, "latency_ms": round(ms, 1)})

    return app


if __name__ == "__main__":
    create_app().run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))

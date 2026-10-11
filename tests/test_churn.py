import json
import time
import unittest

import numpy as np
import pandas as pd

from churn import ab_test, causal
from churn.api import create_app
from churn.data import FRICTION_TPV_EFFECT, generate_panel
from churn.features import build_snapshots
from churn.model import ChurnModel
from churn.smote import smote


class TestChurn(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.panel = generate_panel(n_merchants=3000, seed=1)
        snaps = build_snapshots(cls.panel, range(3, 10))
        cls.train, cls.valid = snaps[snaps.month <= 6], snaps[snaps.month == 7]
        cls.test = snaps[snaps.month.isin([8, 9])]
        cls.model = ChurnModel(n_estimators=120).fit(cls.train, cls.valid)

    def test_churn_is_permanent(self):
        p = self.panel.sort_values(["merchant_id", "month"])
        zero = p["tpv"].eq(0)
        restarted = zero.groupby(p["merchant_id"]).shift(1, fill_value=False) & ~zero
        self.assertFalse(restarted.any())

    def test_no_label_leakage_in_features(self):
        self.assertNotIn("churn", self.model.columns)

    def test_smote_balances_training_only(self):
        X = np.random.default_rng(0).normal(size=(1000, 3))
        y = (np.arange(1000) < 50).astype(int)
        Xs, ys = smote(X, y, ratio=0.5)
        self.assertAlmostEqual(ys.mean(), 1 / 3, places=2)
        self.assertEqual(len(Xs), len(ys))

    def test_model_beats_base_rate(self):
        m = self.model.evaluate(self.test)
        self.assertGreater(m["roc_auc"], 0.70)
        self.assertGreater(m["pr_auc"], 2 * m["base_rate"])

    def test_did_recovers_true_effect(self):
        d = causal.diff_in_diff(self.panel)
        self.assertLess(d["ci95"][0], FRICTION_TPV_EFFECT)
        self.assertGreater(d["ci95"][1], FRICTION_TPV_EFFECT)

    def test_ab_analysis_detects_known_effect(self):
        df = pd.DataFrame({"churn": np.r_[np.ones(800), np.zeros(1200)].astype(int), "tpv": 10_000.0})
        r = ab_test.analyse(ab_test.simulate_outcomes(ab_test.assign(df), relative_reduction=0.4))
        self.assertLess(r["churn_treatment"], r["churn_control"])
        self.assertLess(r["p_value"], 0.05)

    def test_api_scores_fast_and_validates(self):
        client = create_app(self.model).test_client()
        merchant = json.load(open("examples/merchant.json"))
        times = []
        for _ in range(30):
            t0 = time.perf_counter()
            r = client.post("/score", json=merchant)
            times.append((time.perf_counter() - t0) * 1000)
            self.assertEqual(r.status_code, 200)
        self.assertIn(r.get_json()["results"][0]["risk_tier"], {"low", "medium", "high"})
        self.assertLess(np.percentile(times, 95), 200)
        self.assertEqual(client.post("/score", json={"segment": "SMB"}).status_code, 400)


if __name__ == "__main__":
    unittest.main()

import unittest
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import accuracy_score, f1_score

from analysis.metrics import bootstrap, probability_quality, apply_saved_calibration, PCOLS


class AnalysisTests(unittest.TestCase):
    def test_bootstrap_matches_original_sklearn_implementation(self):
        y = np.tile(np.arange(4), 9)
        pred = y.copy(); pred[[1, 4, 10, 21, 30]] = [2, 3, 0, 0, 1]
        frame = pd.DataFrame(dict(target_class=y, predicted_class=pred))
        rng = np.random.default_rng(123)
        acc, f1 = [], []
        for _ in range(100):
            idx = rng.integers(0, len(y), len(y))
            acc.append(accuracy_score(y[idx], pred[idx]))
            f1.append(f1_score(y[idx], pred[idx], labels=range(4), average="macro", zero_division=0))
        result = bootstrap(frame, np.random.default_rng(123), iterations=100)
        np.testing.assert_allclose(result["accuracy_ci"], np.quantile(acc, [.025, .975]), atol=1e-15)
        np.testing.assert_allclose(result["macro_f1_ci"], np.quantile(f1, [.025, .975]), atol=1e-15)

    def test_perfect_probability_quality(self):
        frame = pd.DataFrame(np.eye(4), columns=PCOLS); frame["target_class"] = range(4)
        m = probability_quality(frame)
        self.assertEqual(m["brier"], 0); self.assertEqual(m["ece15"], 0); self.assertEqual(m["accuracy"], 1)
        self.assertLess(m["nll"], 1e-12)

    def test_calibration_uses_supplied_coefficients_without_labels(self):
        ahi = np.array([2., 8.], dtype=np.float64)
        p = np.array([[.7, .2, .09, .01], [.1, .7, .15, .05]])
        parameters = dict(ahi_coefficient=[1.1], ahi_intercept=.02,
            class_coefficients=np.eye(4, 5).tolist(), class_intercepts=[.1, -.1, .2, -.2], classes=[0, 1, 2, 3])
        out_ahi, out_p = apply_saved_calibration(ahi, p, parameters)
        np.testing.assert_allclose(out_ahi, np.expm1(np.log1p(ahi) * 1.1 + .02))
        np.testing.assert_allclose(out_p, softmax(np.log(p) + [.1, -.1, .2, -.2], axis=1))


if __name__ == "__main__": unittest.main()

from collections import deque

import numpy as np
from scipy.stats import chisquare


class DriftMonitor:
    # chi-square test of the predicted class mix in a rolling window vs a reference mix
    def __init__(self, ref_dist, class_names, window_size=200, alpha=0.05, min_samples=50):
        ref = np.asarray(ref_dist, dtype=np.float64)
        self.ref = ref / ref.sum()
        self.class_names = list(class_names)
        self.window = deque(maxlen=window_size)
        self.alpha = alpha
        self.min_samples = min_samples
        self.history = []

    def ingest(self, preds):
        self.window.extend(int(p) for p in preds)
        if len(self.window) < self.min_samples:
            return None
        counts = np.bincount(np.fromiter(self.window, int), minlength=len(self.class_names)).astype(float)
        expected = np.maximum(self.ref * counts.sum(), 1e-6)
        expected *= counts.sum() / expected.sum()
        chi2, p = chisquare(counts, expected)
        contrib = (counts - expected) ** 2 / expected
        res = {
            "step": len(self.history),
            "chi2": float(chi2),
            "p_value": float(p),
            "drift": bool(p < self.alpha),
            "top_classes": [self.class_names[i] for i in np.argsort(contrib)[::-1][:3]],
        }
        self.history.append(res)
        return res

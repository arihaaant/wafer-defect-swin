import numpy as np
import pandas as pd
import pytest


def fake_wafer(rng, kind, size=26):
    yy, xx = np.mgrid[:size, :size]
    r = np.hypot(yy - size / 2, xx - size / 2)
    w = np.where(r < size / 2 - 1, 1, 0)
    if kind == "Center":
        w[r < size / 6] = 2
    elif kind == "Edge-Ring":
        w[(r > size / 2 - 4) & (r < size / 2 - 1)] = 2
    elif kind == "Scratch":
        w[np.abs(yy - xx) < 1.5] = 2
    w[(rng.random(w.shape) < 0.02) & (w == 1)] = 2
    return w


@pytest.fixture
def wafer_df():
    rng = np.random.default_rng(0)
    rows = [{"waferMap": fake_wafer(rng, k), "failureType": k}
            for k in ("Center", "Edge-Ring", "Scratch") for _ in range(20)]
    return pd.DataFrame(rows)

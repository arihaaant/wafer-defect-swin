import json

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from waferdefect.drift import DriftMonitor
from waferdefect.explain import reconstruction_diff, stage_activation_map
from waferdefect.model import FocalLoss, WaferSwin, mixup
from waferdefect.train import Config, train


def test_output_shapes():
    model = WaferSwin(num_classes=8, pretrained=False).eval()
    logits, recon = model(torch.randn(2, 3, 224, 224))
    assert logits.shape == (2, 8)
    assert recon.shape == (2, 1, 224, 224)
    assert 0 <= recon.min() and recon.max() <= 1


def test_focal_loss():
    logits, y = torch.randn(16, 5), torch.randint(0, 5, (16,))
    assert torch.allclose(FocalLoss(gamma=0.0)(logits, y), F.cross_entropy(logits, y))
    assert FocalLoss(gamma=2.0)(logits, y) < F.cross_entropy(logits, y)


def test_mixup():
    torch.manual_seed(0)
    x, y = torch.rand(8, 3, 4, 4), torch.arange(8)
    mixed, y_a, y_b, lam = mixup(x, y, alpha=0.4)
    assert 0 <= lam <= 1
    assert torch.equal(y_a, y)
    assert torch.allclose(mixed, lam * x + (1 - lam) * x[y_b])


def test_drift_monitor():
    rng = np.random.default_rng(0)
    ref = np.array([0.4, 0.3, 0.2, 0.1])

    quiet = DriftMonitor(ref, list("abcd"))
    for _ in range(10):
        quiet.ingest(rng.choice(4, size=50, p=ref))
    assert sum(r["drift"] for r in quiet.history) <= 2

    shifted = DriftMonitor(ref, list("abcd"))
    for _ in range(10):
        shifted.ingest(rng.choice(4, size=50, p=[0.1, 0.1, 0.2, 0.6]))
    assert shifted.history[-1]["drift"]
    assert shifted.history[-1]["top_classes"][0] == "d"


def test_explain_maps():
    model = WaferSwin(num_classes=3, pretrained=False)
    img = torch.randn(3, 224, 224)
    act = stage_activation_map(model, img)
    for a in (act, *reconstruction_diff(model, img)):
        assert a.shape == (224, 224)
    assert 0 <= act.min() and act.max() <= 1


@pytest.mark.parametrize("balanced,weights", [(True, True), (False, False)])
def test_train_smoke(wafer_df, tmp_path, balanced, weights):
    cfg = Config(data="unused", out=str(tmp_path), epochs=1, batch_size=8, num_workers=0,
                 pretrained=False, balanced_sampler=balanced, class_weights=weights)
    metrics = train(cfg, df=wafer_df)
    assert 0 <= metrics["test_macro_f1"] <= 1
    for f in ("best.pt", "config.json", "history.json", "metrics.json"):
        assert (tmp_path / f).exists()
    assert json.loads((tmp_path / "metrics.json").read_text())["class_names"] == ["Center", "Edge-Ring", "Scratch"]

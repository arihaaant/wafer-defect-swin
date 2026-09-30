import numpy as np
import pandas as pd

from waferdefect import data


def test_clean_label():
    assert data.clean_label(np.array([["Scratch"]], dtype=object)) == "Scratch"
    assert data.clean_label(np.array([], dtype=object)) is None
    assert data.clean_label("Loc") == "Loc"


def test_load_keeps_only_labeled_defects(tmp_path):
    df = pd.DataFrame({
        "waferMap": [np.ones((5, 5)), np.ones((5, 5)), np.ones((5, 5)), np.array([])],
        "failureType": [np.array([["Scratch"]]), np.array([["none"]]), np.array([]), np.array([["Loc"]])],
        "dieSize": [1, 2, 3, 4],
    })
    df.to_pickle(tmp_path / "LSWMD.pkl")
    out = data.load_lswmd(str(tmp_path / "LSWMD.pkl"))
    assert out["failureType"].tolist() == ["Scratch"]
    assert list(out.columns) == ["waferMap", "failureType"]


def test_wafer_to_rgb():
    img = data.wafer_to_rgb(np.array([[0, 1], [2, 2]]))
    assert img.shape == (data.img_size, data.img_size, 3)
    assert set(np.unique(img)) == {0, 127, 255}


def test_split(wafer_df):
    train, val, test, names = data.split(wafer_df)
    assert names == ["Center", "Edge-Ring", "Scratch"]
    assert len(train) + len(val) + len(test) == len(wafer_df)
    for part in (train, val, test):
        assert set(part["label"]) == {0, 1, 2}


def test_dataset_item(wafer_df):
    train, *_ = data.split(wafer_df)
    x, y = data.WaferDataset(train)[0]
    assert x.shape == (3, data.img_size, data.img_size)
    assert -1.0 <= x.min() and x.max() <= 1.0
    assert y in (0, 1, 2)

import sys
from collections import Counter

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

img_size = 224
map_col = "waferMap"
label_col = "failureType"


def clean_label(x):
    # labels come as nested arrays like [['Scratch']], or [] when unlabeled
    arr = np.asarray(x, dtype=object).ravel()
    return str(arr[0]) if arr.size else None


def load_lswmd(path):
    # LSWMD.pkl was pickled with an old pandas that had pandas.indexes
    sys.modules.setdefault("pandas.indexes", pd.core.indexes)
    df = pd.read_pickle(path)[[map_col, label_col]].copy()
    df[label_col] = df[label_col].map(clean_label)
    df = df[df[label_col].notna() & (df[label_col] != "none")]
    df = df[df[map_col].map(lambda m: np.asarray(m).size > 0)]
    return df.reset_index(drop=True)


def split(df, seed=42):
    class_names = sorted(df[label_col].unique())
    df = df.assign(label=df[label_col].map({c: i for i, c in enumerate(class_names)}))
    train, rest = train_test_split(df, test_size=0.3, stratify=df["label"], random_state=seed)
    val, test = train_test_split(rest, test_size=0.5, stratify=rest["label"], random_state=seed)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True), class_names


def wafer_to_rgb(wmap):
    # 0 = no die, 1 = good, 2 = defect -> 0, 127, 255
    arr = (np.asarray(wmap, dtype=np.float32) / 2.0 * 255).astype(np.uint8)
    arr = cv2.resize(arr, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
    return np.stack([arr, arr, arr], axis=-1)


normalize = transforms.Normalize([0.5] * 3, [0.5] * 3)

train_tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(90),
    transforms.ToTensor(),
    normalize,
])

eval_tf = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(), normalize])


class WaferDataset(Dataset):
    def __init__(self, df, tf=eval_tf):
        self.maps = df[map_col].tolist()
        self.labels = df["label"].astype(int).tolist()
        self.tf = tf

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.tf(wafer_to_rgb(self.maps[i])), self.labels[i]


def class_counts(df, n):
    counts = Counter(df["label"].tolist())
    return np.array([counts[i] for i in range(n)], dtype=np.float32)


def make_loaders(train, val, test, batch_size=64, num_workers=2, balanced=True):
    kw = dict(batch_size=batch_size, num_workers=num_workers,
              pin_memory=torch.cuda.is_available(), persistent_workers=num_workers > 0)
    if balanced:
        counts = Counter(train["label"].tolist())
        weights = [1.0 / counts[y] for y in train["label"]]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        train_dl = DataLoader(WaferDataset(train, train_tf), sampler=sampler, **kw)
    else:
        train_dl = DataLoader(WaferDataset(train, train_tf), shuffle=True, **kw)
    val_dl = DataLoader(WaferDataset(val), shuffle=False, **kw)
    test_dl = DataLoader(WaferDataset(test), shuffle=False, **kw)
    return train_dl, val_dl, test_dl

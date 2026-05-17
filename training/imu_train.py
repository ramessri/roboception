"""
Trains a 1D CNN on UCI-HAR → 3 macro-classes, exports to ONNX.

Macro-class mapping:
  still       ← SITTING, STANDING, LAYING
  walking     ← WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS
  disturbance ← synthesized: still samples + impulse noise

UCI-HAR has no 'disturbance' class — we synthesize by perturbing still
samples with impulse noise. Defensible: in production we'd collect real
data of phones being bumped/shaken, but the macro behavior (high-variance
spikes) is well-captured by synthesis.

Output:
  models/imu_model.onnx
  models/labels/uci_har_labels.txt
"""
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# ── Paths ─────────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / 'training' / 'data' / 'UCI_HAR_Dataset'
MODEL_OUT = REPO / 'models' / 'imu_model.onnx'
LABELS_OUT = REPO / 'models' / 'labels' / 'uci_har_labels.txt'

MACRO_CLASSES = ['still', 'walking', 'disturbance']

# UCI-HAR raw labels (1-indexed)
# 1=WALKING 2=WALKING_UPSTAIRS 3=WALKING_DOWNSTAIRS 4=SITTING 5=STANDING 6=LAYING
WALKING_LABELS = {1, 2, 3}
STILL_LABELS = {4, 5, 6}

# Tensor shape constants — must match inference node.
WINDOW_LEN = 128
N_CHANNELS = 6   # body_acc xyz + body_gyro xyz

DATASET_URL = ('https://archive.ics.uci.edu/static/public/240/'
               'human+activity+recognition+using+smartphones.zip')

# ── Dataset download ──────────────────────────────────────────────────
def ensure_dataset():
    if DATA_DIR.exists() and (DATA_DIR / 'train').exists():
        print(f'dataset already at {DATA_DIR}')
        return
    DATA_DIR.parent.mkdir(parents=True, exist_ok=True)
    outer_zip = DATA_DIR.parent / 'uci_har_outer.zip'
    print('downloading UCI-HAR (~60 MB)...')
    urllib.request.urlretrieve(DATASET_URL, outer_zip)
    print('extracting...')
    with zipfile.ZipFile(outer_zip) as z:
        z.extractall(DATA_DIR.parent)
    # UCI ships an outer zip containing an inner zip named "UCI HAR Dataset.zip"
    inner_zip = DATA_DIR.parent / 'UCI HAR Dataset.zip'
    if inner_zip.exists():
        with zipfile.ZipFile(inner_zip) as z:
            z.extractall(DATA_DIR.parent)
        spaced = DATA_DIR.parent / 'UCI HAR Dataset'
        if spaced.exists():
            spaced.rename(DATA_DIR)
        inner_zip.unlink()
    outer_zip.unlink()
    print(f'dataset ready at {DATA_DIR}')

# ── Load ──────────────────────────────────────────────────────────────
def load_split(split):
    """split ∈ {'train','test'} → (X: N×128×6, y: N,)"""
    base = DATA_DIR / split / 'Inertial Signals'
    channels = []
    for sensor in ['body_acc', 'body_gyro']:
        for axis in ['x', 'y', 'z']:
            data = np.loadtxt(base / f'{sensor}_{axis}_{split}.txt')   # (N, 128)
            channels.append(data)
    X = np.stack(channels, axis=2).astype(np.float32)                  # (N, 128, 6)
    y = np.loadtxt(DATA_DIR / split / f'y_{split}.txt').astype(np.int64)
    return X, y

def to_macro(y_raw):
    y_macro = np.zeros_like(y_raw)
    y_macro[np.isin(y_raw, list(WALKING_LABELS))] = 1
    y_macro[np.isin(y_raw, list(STILL_LABELS))] = 0
    return y_macro

def synthesize_disturbance(X_still, n, rng):
    """Take still windows, inject impulse spikes + HF noise."""
    idx = rng.choice(len(X_still), size=n, replace=True)
    base = X_still[idx].copy()
    for sample in base:
        n_spikes = rng.integers(3, 9)
        locs = rng.choice(WINDOW_LEN, size=n_spikes, replace=False)
        sample[locs] += rng.normal(0, 3.0, size=(n_spikes, N_CHANNELS))
    base += rng.normal(0, 0.5, size=base.shape).astype(np.float32)
    return base.astype(np.float32)

# ── Model ─────────────────────────────────────────────────────────────
class IMUNet(nn.Module):
    """
    Input: (B, 128, 6) — matches ROS2 inference shape directly.
    Internally transposes to (B, 6, 128) for Conv1d.
    """
    def __init__(self, n_classes=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(N_CHANNELS, 32, 5, padding=2),  nn.BatchNorm1d(32),  nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2),          nn.BatchNorm1d(64),  nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1),         nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        x = x.transpose(1, 2)         # (B,128,6) → (B,6,128)
        return self.head(self.conv(x))

# ── Train ─────────────────────────────────────────────────────────────
def train():
    rng = np.random.default_rng(42)
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    X_tr, y_tr_raw = load_split('train')
    X_te, y_te_raw = load_split('test')
    y_tr = to_macro(y_tr_raw)
    y_te = to_macro(y_te_raw)

    # Synthesize disturbance as ~25% of each split.
    n_dist_tr = len(X_tr) // 4
    n_dist_te = len(X_te) // 4
    X_dist_tr = synthesize_disturbance(X_tr[y_tr == 0], n_dist_tr, rng)
    X_dist_te = synthesize_disturbance(X_te[y_te == 0], n_dist_te, rng)
    X_tr = np.concatenate([X_tr, X_dist_tr])
    y_tr = np.concatenate([y_tr, np.full(n_dist_tr, 2, dtype=np.int64)])
    X_te = np.concatenate([X_te, X_dist_te])
    y_te = np.concatenate([y_te, np.full(n_dist_te, 2, dtype=np.int64)])

    perm = rng.permutation(len(X_tr))
    X_tr, y_tr = X_tr[perm], y_tr[perm]

    print('train counts:', {MACRO_CLASSES[k]: v for k, v in Counter(y_tr).items()})
    print('test  counts:', {MACRO_CLASSES[k]: v for k, v in Counter(y_te).items()})

    train_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    test_ds = TensorDataset(torch.from_numpy(X_te), torch.from_numpy(y_te))
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=64, num_workers=2)

    model = IMUNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    best_acc, best_state = 0.0, None
    for epoch in range(20):
        model.train()
        loss_sum = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
        train_loss = loss_sum / len(train_loader)

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                pred = model(X).argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        acc = correct / total
        print(f'epoch {epoch:2d}  train_loss {train_loss:.4f}  val_acc {acc:.3f}')

        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    print(f'best val acc: {best_acc:.3f}')
    model.load_state_dict(best_state)
    return model

# ── Export ────────────────────────────────────────────────────────────
def export_onnx(model):
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    model.eval().cpu()
    dummy = torch.randn(1, WINDOW_LEN, N_CHANNELS)
    torch.onnx.export(
        model, dummy, str(MODEL_OUT),
        input_names=['imu_window'],
        output_names=['logits'],
        opset_version=12,
        do_constant_folding=True,
    )
    LABELS_OUT.write_text('\n'.join(MACRO_CLASSES) + '\n')
    print(f'exported {MODEL_OUT}')
    print(f'labels   {LABELS_OUT}')

def main():
    ensure_dataset()
    model = train()
    export_onnx(model)

if __name__ == '__main__':
    main()
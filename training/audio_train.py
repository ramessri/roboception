"""
Trains a small CNN on ESC-50 (subset → 4 macro-classes) and exports to ONNX.

Output:
  models/audio_model.onnx                  — the deployable model
  models/labels/esc50_macro_labels.txt     — 4 macro-class names, one per line
"""
import os
import random
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchaudio.transforms as T
import torchaudio.functional
import soundfile as sf
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# ── Paths ─────────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / 'training' / 'data' / 'ESC-50-master'
MODEL_OUT = REPO / 'models' / 'audio_model.onnx'
LABELS_OUT = REPO / 'models' / 'labels' / 'esc50_macro_labels.txt'

# ── Macro-class mapping ───────────────────────────────────────────────
MACRO_CLASSES = ['silence', 'human', 'appliance', 'alert']

ESC50_TO_MACRO = {
    # human (non-speech vocalizations + body sounds)
    'crying_baby': 'human', 'sneezing': 'human', 'clapping': 'human',
    'breathing': 'human', 'coughing': 'human', 'footsteps': 'human',
    'laughing': 'human', 'brushing_teeth': 'human', 'snoring': 'human',
    'drinking_sipping': 'human',
    # appliance (mechanical/domestic)
    'vacuum_cleaner': 'appliance', 'clock_alarm': 'appliance',
    'clock_tick': 'appliance', 'washing_machine': 'appliance',
    'can_opening': 'appliance', 'keyboard_typing': 'appliance',
    'mouse_click': 'appliance', 'door_wood_knock': 'appliance',
    'door_wood_creaks': 'appliance',
    # alert (loud, attention-demanding)
    'siren': 'alert', 'car_horn': 'alert', 'engine': 'alert',
    'train': 'alert', 'church_bells': 'alert', 'fireworks': 'alert',
    'hand_saw': 'alert', 'chainsaw': 'alert', 'glass_breaking': 'alert',
}
# 'silence' has no ESC-50 source — we synthesize.

LABEL_TO_IDX = {label: idx for idx, label in enumerate(MACRO_CLASSES)}

# ── Audio preprocessing constants ─────────────────────────────────────
SAMPLE_RATE = 22050
WINDOW_SAMPLES = SAMPLE_RATE       # 1 second
N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 64
TIME_FRAMES = 44                   # mel-spec time dimension after trim/pad

# ── Dataset download ──────────────────────────────────────────────────
DATASET_URL = 'https://github.com/karoldvl/ESC-50/archive/master.zip'

def ensure_dataset():
    if DATA_DIR.exists() and (DATA_DIR / 'meta' / 'esc50.csv').exists():
        print(f'dataset already at {DATA_DIR}')
        return
    DATA_DIR.parent.mkdir(parents=True, exist_ok=True)
    zip_path = DATA_DIR.parent / 'esc50.zip'
    print(f'downloading ESC-50 (~600 MB) → {zip_path}')
    urllib.request.urlretrieve(DATASET_URL, zip_path)
    print('extracting...')
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(DATA_DIR.parent)
    extracted = DATA_DIR.parent / 'ESC-50-master'
    if not extracted.exists():
        # GitHub zips sometimes nest the dir name; find and rename
        candidates = list(DATA_DIR.parent.glob('ESC-50*'))
        if candidates:
            candidates[0].rename(DATA_DIR)
    zip_path.unlink()
    print(f'dataset ready at {DATA_DIR}')

# ── Preprocessing pipeline ────────────────────────────────────────────
mel_transform = nn.Sequential(
    T.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
    ),
    T.AmplitudeToDB(),
)

def preprocess_waveform(waveform: torch.Tensor, orig_sr: int) -> torch.Tensor:
    """Raw audio (1, N) at orig_sr → mel-spec (1, 64, 44)."""
    if waveform.dim() == 2 and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)  # stereo → mono

    if orig_sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, orig_sr, SAMPLE_RATE)

    # Center-crop or pad to exactly 1 second
    n = waveform.shape[1]
    if n > WINDOW_SAMPLES:
        start = (n - WINDOW_SAMPLES) // 2
        waveform = waveform[:, start:start + WINDOW_SAMPLES]
    elif n < WINDOW_SAMPLES:
        waveform = torch.nn.functional.pad(waveform, (0, WINDOW_SAMPLES - n))

    mel = mel_transform(waveform)                      # (1, 64, T)
    if mel.shape[2] > TIME_FRAMES:
        mel = mel[:, :, :TIME_FRAMES]
    elif mel.shape[2] < TIME_FRAMES:
        mel = torch.nn.functional.pad(mel, (0, TIME_FRAMES - mel.shape[2]))

    mel = (mel - mel.mean()) / (mel.std() + 1e-6)
    return mel                                         # (1, 64, 44)

# ── Dataset class ─────────────────────────────────────────────────────
class ESC50Dataset(Dataset):
    """
    ESC-50 clips filtered to our chosen subset, plus synthetic silence.

    folds: which ESC-50 folds (1-5) to include. Use [1,2,3,4] for train,
           [5] for val. Keeps source-recording leakage out of validation.
    """
    def __init__(self, folds, n_silence=200):
        meta = pd.read_csv(DATA_DIR / 'meta' / 'esc50.csv')
        meta = meta[meta['fold'].isin(folds)]
        meta = meta[meta['category'].isin(ESC50_TO_MACRO)]

        self.items = []
        for _, row in meta.iterrows():
            wav_path = DATA_DIR / 'audio' / row['filename']
            macro = ESC50_TO_MACRO[row['category']]
            self.items.append(('file', str(wav_path), LABEL_TO_IDX[macro]))

        # Add synthetic silence clips.
        for _ in range(n_silence):
            self.items.append(('silence', None, LABEL_TO_IDX['silence']))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        kind, path, label = self.items[idx]
        if kind == 'silence':
            waveform = 0.001 * torch.randn(1, WINDOW_SAMPLES)
            sr = SAMPLE_RATE
        else:
            data, sr = sf.read(path, dtype='float32')  # (N,) or (N, C)
            waveform = torch.from_numpy(data)
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)       # (1, N)
            else:
                waveform = waveform.T                   # (C, N)
        mel = preprocess_waveform(waveform, sr)
        return mel, label

# ── Model ─────────────────────────────────────────────────────────────
class AudioCNN(nn.Module):
    def __init__(self, n_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),   nn.BatchNorm2d(32),  nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),  nn.BatchNorm2d(64),  nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, padding=1),nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# ── Training loop ─────────────────────────────────────────────────────
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    train_ds = ESC50Dataset(folds=[1, 2, 3, 4], n_silence=200)
    val_ds = ESC50Dataset(folds=[5], n_silence=50)

    # Class balance check.
    from collections import Counter
    train_counts = Counter(item[2] for item in train_ds.items)
    print('train class counts:', {MACRO_CLASSES[k]: v for k, v in train_counts.items()})

    # Weighted sampler — inverse-frequency weight per sample.
    sample_weights = [1.0 / train_counts[item[2]] for item in train_ds.items]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ds), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=32, sampler=sampler, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2)

    model = AudioCNN(n_classes=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state = None

    for epoch in range(30):
        model.train()
        train_loss = 0.0
        for mel, label in train_loader:
            mel, label = mel.to(device), label.to(device)
            optimizer.zero_grad()
            logits = model(mel)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for mel, label in val_loader:
                mel, label = mel.to(device), label.to(device)
                pred = model(mel).argmax(dim=1)
                correct += (pred == label).sum().item()
                total += label.size(0)
        val_acc = correct / total

        scheduler.step()
        print(f'epoch {epoch:2d}  train_loss {train_loss:.4f}  val_acc {val_acc:.3f}')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    print(f'best val acc: {best_val_acc:.3f}')
    model.load_state_dict(best_state)
    return model

# ── ONNX export ───────────────────────────────────────────────────────
def export_onnx(model):
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    model.eval().cpu()
    dummy = torch.randn(1, 1, N_MELS, TIME_FRAMES)
    torch.onnx.export(
        model, dummy, str(MODEL_OUT),
        input_names=['mel_spectrogram'],
        output_names=['logits'],
        opset_version=12,
        do_constant_folding=True,
    )
    LABELS_OUT.write_text('\n'.join(MACRO_CLASSES) + '\n')
    print(f'exported {MODEL_OUT}')
    print(f'labels   {LABELS_OUT}')

# ── Main ──────────────────────────────────────────────────────────────
def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    ensure_dataset()
    model = train()
    export_onnx(model)

if __name__ == '__main__':
    main()
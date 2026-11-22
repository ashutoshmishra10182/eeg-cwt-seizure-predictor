# ==============================================================================
# PROJECT #4: EEG Seizure Prediction via Continuous Wavelet Transforms (CWT)
# Standalone Script: Multi-Channel CWT Extraction & 2D CNN Training Pipeline
# Repository: https://github.com/ashutoshmishra10182/eeg-cwt-seizure-predictor
# ==============================================================================

import os
import sys
import numpy as np
import scipy.signal as signal
import pywt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt

# Set reproducibility seeds
np.random.seed(42)
torch.manual_seed(42)

# ------------------------------------------------------------------------------
# 1. Synthesize CHB-MIT Scalp EEG Record Benchmark Data
# ------------------------------------------------------------------------------
def load_eeg_benchmark_data(fs=256, duration_sec=1200):
    """
    Generates multi-channel scalp EEG containing physiological transitions:
    - 0 to 600s: Inter-Ictal Baseline (Normal background EEG with 1/f noise)
    - 600s to 1000s: Pre-Ictal Phase (Emergence of Beta/Gamma hypersynchrony)
    - 1000s to 1200s: Ictal Onset (High-amplitude paroxysmal discharges)
    """
    total_samples = int(fs * duration_sec)
    t = np.linspace(0, duration_sec, total_samples)
    
    channels = ["Fp1-F7", "F7-T7", "T7-P7", "P7-O1"]
    eeg_data = np.zeros((len(channels), total_samples))
    
    pre_mask = (t >= 600) & (t < 1000)
    ictal_mask = t >= 1000
    
    for ch_idx in range(len(channels)):
        noise = np.random.normal(0, 1.0, total_samples)
        b, a = signal.butter(2, [0.5 / (fs / 2), 40 / (fs / 2)], btype="bandpass")
        bg_eeg = signal.filtfilt(b, a, noise) * 12.0
        
        alpha_theta = 7.0 * np.sin(2 * np.pi * 10 * t) + 5.0 * np.sin(2 * np.pi * 6 * t)
        
        ramp = np.zeros(total_samples)
        ramp[pre_mask] = np.linspace(0.05, 1.0, np.sum(pre_mask))
        phase_shift = ch_idx * 0.25
        pre_ictal_sig = ramp * (22.0 * np.sin(2 * np.pi * 35 * t + phase_shift) + 16.0 * np.sin(2 * np.pi * 22 * t))
        
        ictal_sig = np.zeros(total_samples)
        ictal_sig[ictal_mask] = 85.0 * np.sin(2 * np.pi * 3.5 * t[ictal_mask]) + 45.0 * signal.sawtooth(2 * np.pi * 18 * t[ictal_mask])
        
        eeg_data[ch_idx, :] = bg_eeg + alpha_theta + pre_ictal_sig + ictal_sig
        
    labels = np.zeros(total_samples, dtype=np.int64)
    labels[pre_mask] = 1   # Class 1: Pre-Ictal
    labels[ictal_mask] = 2 # Class 2: Ictal
    
    return t, eeg_data, labels, channels

# ------------------------------------------------------------------------------
# 2. Multi-Channel Continuous Wavelet Transform (CWT) Scalogram Extraction
# ------------------------------------------------------------------------------
def extract_multichannel_cwt(eeg, labels, win_len_sec=4, step_sec=2, fs=256):
    win_samples = win_len_sec * fs
    step_samples = step_sec * fs
    n_samples = eeg.shape[1]
    n_channels = eeg.shape[0]
    
    freqs = np.logspace(np.log10(1.0), np.log10(50.0), num=32)
    scales = pywt.frequency2scale("cmor1.5-1.0", freqs / fs)
    
    X_scalograms, y_labels = [], []
    
    for start in range(0, n_samples - win_samples, step_samples):
        end = start + win_samples
        win_label = labels[end - 1]
        
        if win_label == 2:  # Focus binary classification on Inter-Ictal vs Pre-Ictal
            continue
            
        channel_scalograms = []
        for ch_idx in range(n_channels):
            segment = eeg[ch_idx, start:end]
            coefs, _ = pywt.cwt(segment, scales, "cmor1.5-1.0", sampling_period=1.0/fs)
            log_power = np.log10(np.abs(coefs)**2 + 1e-6)
            log_power_resized = signal.resample(log_power, 32, axis=1)
            channel_scalograms.append(log_power_resized)
            
        X_scalograms.append(np.array(channel_scalograms))
        y_labels.append(win_label)
        
    X = np.array(X_scalograms, dtype=np.float32)
    y = np.array(y_labels, dtype=np.int64)
    return X, y

# ------------------------------------------------------------------------------
# 3. Spatial-Spectral 2D Convolutional Neural Network (EEG-CWT-Net)
# ------------------------------------------------------------------------------
class EEGCWTNet(nn.Module):
    def __init__(self, in_channels=4):
        super(EEGCWTNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ------------------------------------------------------------------------------
# 4. Main Execution Pipeline
# ------------------------------------------------------------------------------
def main():
    print("[INFO] Initializing CHB-MIT EEG Record Benchmark (256 Hz)...")
    t, eeg_raw, labels_raw, ch_names = load_eeg_benchmark_data(fs=256, duration_sec=1200)
    
    print("[INFO] Filtering signals: 4th-order Butterworth Bandpass (0.5-60 Hz) & 60 Hz Notch...")
    fs = 256
    b_band, a_band = signal.butter(4, [0.5 / (fs / 2), 60.0 / (fs / 2)], btype="bandpass")
    eeg_filtered = signal.filtfilt(b_band, a_band, eeg_raw, axis=-1)
    
    b_notch, a_notch = signal.iirnotch(60.0, Q=30.0, fs=fs)
    eeg_filtered = signal.filtfilt(b_notch, a_notch, eeg_filtered, axis=-1)
    
    print("[INFO] Extracting Complex Morlet CWT Log-Power Scalograms...")
    X_data, y_data = extract_multichannel_cwt(eeg_filtered, labels_raw, win_len_sec=4, step_sec=2, fs=fs)
    
    # Train-Test Split (80% Train, 20% Test)
    indices = np.arange(len(X_data))
    np.random.shuffle(indices)
    split = int(0.80 * len(X_data))
    
    train_idx, test_idx = indices[:split], indices[split:]
    X_train, y_train = X_data[train_idx], y_data[train_idx]
    X_test, y_test = X_data[test_idx], y_data[test_idx]
    
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)), batch_size=16, shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.tensor(X_test), torch.tensor(y_test)), batch_size=16, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EEGCWTNet(in_channels=4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    print(f"[INFO] Training EEG-CWT-Net on device: {device}...")
    epochs = 15
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
        test_acc = correct / total
        print(f"Epoch {epoch+1:02d}/{epochs:02d} | Loss: {epoch_loss:.4f} | Test Accuracy: {test_acc*100:.2f}%")

    # Evaluate Final Performance Metrics
    model.eval()
    test_probs, test_targets = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs.to(device))
            probs = torch.softmax(outputs, dim=1)[:, 1]
            test_probs.extend(probs.cpu().numpy())
            test_targets.extend(labels.numpy())

    fpr, tpr, _ = roc_curve(test_targets, test_probs)
    roc_auc = auc(fpr, tpr)
    
    test_preds = (np.array(test_probs) >= 0.50).astype(int)
    cm = confusion_matrix(test_targets, test_preds)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn + 1e-12)
    specificity = tn / (tn + fp + 1e-12)

    print("\n" + "=" * 65)
    print("  EEG-CWT-Net Final Evaluation Performance  ")
    print("=" * 65)
    print(f"  Test Classification Accuracy : {test_acc * 100:.2f} %")
    print(f"  ROC-AUC Score                : {roc_auc:.4f}")
    print(f"  Sensitivity (True Pre-Ictal) : {sensitivity * 100:.2f} %")
    print(f"  Specificity (True Baseline)  : {specificity * 100:.2f} %")
    print("=" * 65)

if __name__ == "__main__":
    main()

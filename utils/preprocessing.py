"""
EEG preprocessing utilities for DEAP-based feature extraction.

Includes:
 - 3s baseline removal (keeps only 60s stimulus EEG)
 - Butterworth bandpass filtering (4-45 Hz, order=5) - can be used in feature extraction
 - Per-channel z-score normalization
 - Epoching trials into 8s windows with 4s step (50% overlap)
 - Mapping valence & arousal to binary (0 = low, 1 = high)
"""

import numpy as np
from scipy.signal import butter, filtfilt

# ==========================================================
# 1️⃣ BANDPASS FILTER (for per-band filtering in feature extraction)
# ==========================================================
def bandpass_filter(signal, lowcut=4, highcut=45, fs=128, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal)

# ==========================================================
# 2️⃣ BASELINE REMOVAL
# ==========================================================
def remove_baseline(eeg_data, baseline_duration=3, fs=128):
    baseline_samples = baseline_duration * fs
    if eeg_data.ndim == 3:
        return eeg_data[:, :, baseline_samples:]
    elif eeg_data.ndim == 2:
        return eeg_data[:, baseline_samples:]
    else:
        raise ValueError(f"Expected 2D or 3D array, got {eeg_data.ndim}D")

# ==========================================================
# 3️⃣ EPOCHING FUNCTION
# ==========================================================
def epoch_signal(trial_signal, window_size=1024, step_size=512, fs=128):
    channels, total_samples = trial_signal.shape
    num_epochs = (total_samples - window_size) // step_size + 1
    epochs = np.zeros((num_epochs, channels, window_size), dtype=np.float32)
    for i in range(num_epochs):
        start = i * step_size
        end = start + window_size
        epochs[i] = trial_signal[:, start:end]
    return epochs

# ==========================================================
# 4️⃣ BINARY MAPPING (Valence & Arousal)
# ==========================================================
def map_valence_arousal(valence, arousal, threshold=5.0):
    v_label = 1 if valence >= threshold else 0
    a_label = 1 if arousal >= threshold else 0
    return np.array([v_label, a_label], dtype=np.int32)

# ==========================================================
# 5️⃣ FULL SUBJECT PIPELINE
# ==========================================================
def preprocess_and_epoch_subject(
    eeg_data, labels, normalize=True,
    remove_baseline_period=True, window_size=1024, step_size=512
):
    if remove_baseline_period:
        eeg_data = remove_baseline(eeg_data, baseline_duration=3, fs=128)

    all_epochs, all_labels = [], []

    for trial_idx in range(len(eeg_data)):
        trial = eeg_data[trial_idx, :32, :].copy()  # EEG channels only

        if normalize:
            for ch in range(32):
                sig = trial[ch]
                trial[ch] = (sig - np.mean(sig)) / (np.std(sig) + 1e-8)

        trial_epochs = epoch_signal(trial, window_size, step_size, fs=128)
        all_epochs.append(trial_epochs)

        v_label, a_label = map_valence_arousal(*labels[trial_idx][:2])
        label_pair = np.tile([v_label, a_label], (trial_epochs.shape[0], 1))
        all_labels.append(label_pair)

    X_epochs = np.vstack(all_epochs).astype(np.float32)
    Y_labels = np.vstack(all_labels).astype(np.int32)
    return X_epochs, Y_labels

# ==========================================================
# 6️⃣ INFO HELPER
# ==========================================================
def get_preprocessing_info():
    return {
        "baseline_removal": "3s (384 samples @ 128Hz)",
        "bandpass_filter": "Butterworth 4–45Hz (order=5)",
        "normalization": "Per-channel z-score",
        "epoching": "8s window, 4s step (50% overlap)",
        "epochs_per_trial": 15,
        "labels": "binary valence & arousal (0=low, 1=high)",
    }

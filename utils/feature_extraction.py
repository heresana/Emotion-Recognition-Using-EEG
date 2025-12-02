"""
utils/feature_extraction.py
------------------------------------
Differential Entropy (DE) + additional EEG features extractor for DEAP dataset.

✔ Consistent bandpass_filter (order=5) with preprocessing.py
✔ No across-channel normalization
✔ 8s window, 4s step (50% overlap)
✔ Per-channel normalization only
✔ Returns feature tensor: (n_epochs, n_channels, n_features)

Feature breakdown per channel (for 4 bands):
    - DE (log variance): 4
    - Band Power (log FFT): 4
    - Hjorth: 3
    - Time-domain: 6
Total per-channel features = 17
"""

import numpy as np
from scipy.stats import skew, kurtosis

# Import preprocessing utilities
try:
    from utils.preprocessing import bandpass_filter, epoch_signal
except ImportError:
    from preprocessing import bandpass_filter, epoch_signal  # fallback if relative import fails

# Frequency bands (DEAP standard)
BANDS = {
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45),
}

# ==========================================================
# 1️⃣ DIFFERENTIAL ENTROPY (log variance)
# ==========================================================
def compute_de_for_epoch(epoch, fs=128, bands=BANDS):
    n_channels = epoch.shape[0]
    out = np.zeros((n_channels, len(bands)), dtype=np.float32)
    eps = 1e-10

    for b_idx, (bname, (low, high)) in enumerate(bands.items()):
        for ch in range(n_channels):
            sig = epoch[ch]
            filtered = bandpass_filter(sig, lowcut=low, highcut=high, fs=fs, order=5)
            var = np.var(filtered)
            out[ch, b_idx] = np.log(var + eps)
    return out

# ==========================================================
# 2️⃣ BAND POWER (FFT-based log power)
# ==========================================================
def _band_power_fft(epoch, fs=128, bands=BANDS):
    n_channels, n_samples = epoch.shape
    out = np.zeros((n_channels, len(bands)), dtype=np.float32)
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / fs)
    eps = 1e-12

    for ch in range(n_channels):
        sig = epoch[ch]
        fft_vals = np.abs(np.fft.rfft(sig)) ** 2
        for b_idx, (bname, (low, high)) in enumerate(bands.items()):
            idx = np.where((freqs >= low) & (freqs <= high))[0]
            if len(idx) == 0:
                continue
            band_power = np.sum(fft_vals[idx]) / (len(idx) + eps)
            out[ch, b_idx] = np.log(band_power + eps)
    return out

# ==========================================================
# 3️⃣ HJORTH PARAMETERS (per channel)
# ==========================================================
def _hjorth_params(epoch):
    n_channels, _ = epoch.shape
    out = np.zeros((n_channels, 3), dtype=np.float32)
    eps = 1e-12

    for ch in range(n_channels):
        x = epoch[ch]
        dx = np.diff(x)
        ddx = np.diff(dx)
        var_x = np.var(x)
        var_dx = np.var(dx)
        var_ddx = np.var(ddx) if ddx.size > 0 else 0.0

        activity = var_x
        mobility = np.sqrt(var_dx / (var_x + eps))
        mobility_dx = np.sqrt(var_ddx / (var_dx + eps)) if var_dx > 0 else 0.0
        complexity = mobility_dx / (mobility + eps)
        out[ch] = [activity, mobility, complexity]

    # Per-channel normalization
    for ch in range(n_channels):
        mean_val = out[ch].mean()
        std_val = out[ch].std()
        out[ch] = (out[ch] - mean_val) / (std_val + eps)
    return out

# ==========================================================
# 4️⃣ TIME-DOMAIN FEATURES
# ==========================================================
def _time_domain_features(epoch):
    n_channels, _ = epoch.shape
    out = np.zeros((n_channels, 6), dtype=np.float32)
    eps = 1e-12

    for ch in range(n_channels):
        sig = epoch[ch]
        mean_val = np.mean(sig)
        std_val = np.std(sig)
        skew_val = skew(sig)
        kurt_val = kurtosis(sig)
        zcr = ((sig[:-1] * sig[1:]) < 0).sum() / (len(sig) - 1)
        energy = np.sum(sig**2) / len(sig)
        out[ch] = [mean_val, std_val, skew_val, kurt_val, zcr, energy]

    # Per-channel normalization
    for ch in range(n_channels):
        mean_val = out[ch].mean()
        std_val = out[ch].std()
        out[ch] = (out[ch] - mean_val) / (std_val + eps)
    return out

# ==========================================================
# 5️⃣ TRIAL → EPOCH FEATURES
# ==========================================================
def trial_to_de_epochs(trial, fs=128, window_size=1024, step_size=512, bands=BANDS):
    epochs = epoch_signal(trial, window_size=window_size, step_size=step_size)
    n_epochs, n_channels = epochs.shape[0], epochs.shape[1]
    n_feats = len(bands) * 2 + 3 + 6  # 17 features/channel

    out = np.zeros((n_epochs, n_channels, n_feats), dtype=np.float32)
    for i in range(n_epochs):
        ep = epochs[i]
        de_feats = compute_de_for_epoch(ep, fs=fs, bands=bands)
        bp_feats = _band_power_fft(ep, fs=fs, bands=bands)
        hjorth_feats = _hjorth_params(ep)
        td_feats = _time_domain_features(ep)
        out[i] = np.concatenate([de_feats, bp_feats, hjorth_feats, td_feats], axis=1)
    return out

# ==========================================================
# 6️⃣ BATCH EXTRACTION FOR SUBJECT
# ==========================================================
# def extract_features_for_subject(preprocessed_trials, fs=128, window_size=1024, step_size=512, bands=BANDS):
#     all_epoch_features = []
#     for trial_idx in range(preprocessed_trials.shape[0]):
#         trial = preprocessed_trials[trial_idx]
#         trial_features = trial_to_de_epochs(
#             trial, fs=fs, window_size=window_size, step_size=step_size, bands=bands
#         )
#         all_epoch_features.append(trial_features)
#     features = np.vstack(all_epoch_features).astype(np.float32)
#     return features


# ==========================================================
# 6️⃣ BATCH EXTRACTION FOR SUBJECT (NO RE-EPOCHING)
# ==========================================================
def extract_features_for_subject(preprocessed_epochs, fs=128, bands=BANDS):
    """
    preprocessed_epochs: already epoched data (n_epochs, 32, 1024)
    Extract DE + FFT + Hjorth + Time features per epoch.
    """
    all_features = []

    for ep in preprocessed_epochs:   # ep shape = (32, 1024)
        de  = compute_de_for_epoch(ep, fs=fs, bands=bands)
        bp  = _band_power_fft(ep, fs=fs, bands=bands)
        hj  = _hjorth_params(ep)
        td  = _time_domain_features(ep)

        feats = np.concatenate([de, bp, hj, td], axis=1)  # (32, 17)
        all_features.append(feats)

    return np.array(all_features, dtype=np.float32)       # (n_epochs, 32, 17)


# ==========================================================
# 7️⃣ FEATURE INFO
# ==========================================================
def get_feature_info():
    return {
        "bands": BANDS,
        "n_bands": len(BANDS),
        "features_per_channel": {
            "DE (log variance)": len(BANDS),
            "Band Power (log FFT)": len(BANDS),
            "Hjorth parameters": 3,
            "Time-domain": 6,
        },
        "total_features_per_channel": 17,
        "window_size": "8s (1024 samples @ 128 Hz)",
        "step_size": "4s (512 samples @ 128 Hz)",
        "overlap": "50%",
        "epochs_per_trial": 14,
        "epochs_per_subject":560,
        "filter_order": 5,
        "sampling_rate": 128,
        "n_eeg_channels": 32,
    }

# """
# DEAP Preprocessing + Feature Extraction + Save per subject
# - 30 subjects → train/val (no prefix)
# - 2 subjects → test/demo (prefix: 'test_')
# - Saves per-subject .npy in 'processed' folder
# - Uses utils/feature_extraction.py
# """

# import os,sys
# import numpy as np

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# from utils.preprocessing import preprocess_and_epoch_subject
# from utils.feature_extraction import extract_features_for_subject

# DATA_PATH = "data/raw/data_preprocessed_python"
# PROCESSED_PATH = "data/processed"
# os.makedirs(PROCESSED_PATH, exist_ok=True)

# SUBJECT_IDS = list(range(1, 33))  # 32 subjects
# TRAIN_VAL_SUBJECTS = SUBJECT_IDS[:30]
# TEST_SUBJECTS = SUBJECT_IDS[30:]  # last 2 subjects

# def load_subject_data(subject_id):
#     file_path = os.path.join(DATA_PATH, f"s{subject_id:02d}.dat")
#     data = np.load(file_path, allow_pickle=True)
    
#     if isinstance(data, np.ndarray) and data.shape == ():
#         data = data.item() 

#     eeg_data = data['data']  # (40 trials, 40 channels, 8064 samples)
#     labels = data['labels']  # (40 trials, 4 ratings)
#     return eeg_data, labels

# def preprocess_extract_save(subject_ids, prefix=""):
#     """
#     Preprocess subjects → extract features → save X/Y per subject.
#     prefix: optional, e.g., 'test_' for test subjects
#     """
#     for sub_id in subject_ids:
#         print(f"\nProcessing subject {sub_id:02d}...")
#         eeg_data, labels = load_subject_data(sub_id)

#         # 1️⃣ Preprocess + epoch
#         X_epochs, Y_labels = preprocess_and_epoch_subject(
#             eeg_data, labels, normalize=True
#         )  # X_epochs: (n_epochs, 32, 1024), Y_labels: (n_epochs, 2)

#         # 2️⃣ Feature extraction
#         X_features = extract_features_for_subject(X_epochs)  # (n_epochs, 32, 17)

#         # 3️⃣ Save per-subject
#         x_file = os.path.join(PROCESSED_PATH, f"{prefix}s{sub_id:02d}_X.npy")
#         y_file = os.path.join(PROCESSED_PATH, f"{prefix}s{sub_id:02d}_Y.npy")
#         np.save(x_file, X_features)
#         np.save(y_file, Y_labels)

#         print(f"Saved {prefix or 'train/val'} subject {sub_id:02d} -> X: {X_features.shape}, Y: {Y_labels.shape}")

# # ===================== TRAIN/VAL =====================
# preprocess_extract_save(TRAIN_VAL_SUBJECTS)  # no prefix

# # ===================== TEST/DEMO =====================
# preprocess_extract_save(TEST_SUBJECTS, prefix="test_")

"""
01_load_deap.py
-------------------------
• Loads DEAP .dat files 
• Preprocesses + epochs using utils.preprocessing.preprocess_and_epoch_subject()
• Extracts features per-epoch using utils.feature_extraction.extract_features_for_subject()
• Converts SAM ratings → 
      - Binary Valence (0=Low,1=High)
      - Binary Arousal (0=Low,1=High)
      - Russell Circumplex Quadrants (0–3)
• Saves per-subject X and Y (Y columns = [val_bin, aro_bin, quadrant])
• Ensures quadrant repetition count exactly matches epochs produced by preprocessing
"""

import os
import sys
import pickle
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# utils
from utils.preprocessing import (
    preprocess_and_epoch_subject,
    epoch_signal,
    remove_baseline,
)
from utils.feature_extraction import extract_features_for_subject

# ---------------- PATHS ----------------
DATA_PATH = os.path.join("data", "raw", "data_preprocessed_python")
SAVE_PATH = os.path.join("data", "processed")
os.makedirs(SAVE_PATH, exist_ok=True)

# -------- SUBJECT SPLIT ----------
SUBJECTS = list(range(1, 33))
TRAIN_VAL_SUBJECTS = SUBJECTS[:30]   # 1–30
TEST_SUBJECTS = SUBJECTS[30:]        # 31–32

# ---------------- LABEL MAPPING ----------------

def map_binary(x):
    """1–9 SAM → binary."""
    return 1 if x >= 5 else 0

def map_circumplex(val, aro):
    """4-quadrant Russell model."""
    v = map_binary(val)
    a = map_binary(aro)
    if v == 0 and a == 0:
        return 0  # sad/depressed
    elif v == 0 and a == 1:
        return 1  # angry/stressed
    elif v == 1 and a == 0:
        return 2  # calm/relaxed
    else:
        return 3  # happy/excited

# ---------------- MAIN PIPELINE ----------------

def load_subject_data(subject_id):
    file_path = os.path.join(DATA_PATH, f"s{subject_id:02d}.dat")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Missing file: {file_path}")
    with open(file_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    eeg_data = data["data"]      # (40, 40, 8064)
    labels = data["labels"]      # (40, 4)
    return eeg_data, labels


def process_subject(subject_id, prefix=""):
    print(f"\nProcessing Subject {subject_id:02d} ({'TEST' if prefix else 'TRAIN'})")

    eeg_data, labels = load_subject_data(subject_id)

    # ---------------- 1) Preprocess + Epoch (8s window, 4s step) --------------
    # This returns stacked epochs across all trials and binary labels per epoch
    X_epochs, Y_bin = preprocess_and_epoch_subject(
        eeg_data,
        labels,
        normalize=True,
        remove_baseline_period=True,
        window_size=1024,   # 8 seconds @128Hz
        step_size=512       # 4 seconds step
    )
    # X_epochs: (n_epochs, 32, 1024)
    # Y_bin: (n_epochs, 2)  → [val_bin, aro_bin]

    # ---------------- 1.5) Compute per-trial epoch counts (so quadrants repeat correctly)
    # We must replicate the exact steps used in preprocess_and_epoch_subject:
    per_trial_epoch_counts = []
    for trial_idx in range(len(eeg_data)):
        # Take EEG channels only and remove baseline similarly
        trial = eeg_data[trial_idx, :32, :].copy()  # (32, samples)

        # remove baseline (same as preprocess uses)
        trial = remove_baseline(trial, baseline_duration=3, fs=128)

        # normalize per-channel in same way
        for ch in range(trial.shape[0]):
            sig = trial[ch]
            trial[ch] = (sig - np.mean(sig)) / (np.std(sig) + 1e-8)

        # epoch this single trial using same window/step
        trial_epochs = epoch_signal(trial, window_size=1024, step_size=512, fs=128)
        per_trial_epoch_counts.append(trial_epochs.shape[0])

    total_count_from_trials = sum(per_trial_epoch_counts)
    if total_count_from_trials != X_epochs.shape[0]:
        # something odd happened — warn with helpful diagnostics
        raise RuntimeError(
            f"Epoch count mismatch for subject {subject_id}: "
            f"sum(per-trial epochs)={total_count_from_trials} != preprocessed X_epochs rows={X_epochs.shape[0]}.\n"
            "This indicates preprocessing and local epoch counting diverged. "
            "Please check preprocess_and_epoch_subject parameters and data integrity."
        )

    # ---------------- 2) Extract 17 EEG features per epoch -----------------
    X_features = extract_features_for_subject(X_epochs)
    # X_features: (n_epochs, 32, 17)

    # ---------------- 3) Add Russell Quadrant Mapping ------------------------
    quadrants = []
    for t in range(len(labels)):   # 40 trials
        val = labels[t, 0]
        aro = labels[t, 1]
        quad = map_circumplex(val, aro)

        # repeat quad for the exact number of epochs that trial produced
        rep_count = per_trial_epoch_counts[t]
        quadrants.extend([quad] * rep_count)

    quadrants = np.array(quadrants, dtype=np.int32).reshape(-1, 1)  # (n_epochs, 1)

    # ---------------- 4) Final Y = [val_bin, aro_bin, quadrant] --------------
    # Ensure shapes match, otherwise raise informative error
    if Y_bin.shape[0] != quadrants.shape[0] or X_features.shape[0] != quadrants.shape[0]:
        raise RuntimeError(
            f"Shape mismatch after feature extraction for subject {subject_id}: "
            f"X_features={X_features.shape}, Y_bin={Y_bin.shape}, quadrants={quadrants.shape}"
        )

    Y_final = np.hstack([Y_bin, quadrants])  # (n_epochs, 3)

    # ---------------- 5) Save outputs ----------------------------------------
    x_file = os.path.join(SAVE_PATH, f"{prefix}s{subject_id:02d}_X.npy")
    y_file = os.path.join(SAVE_PATH, f"{prefix}s{subject_id:02d}_Y.npy")
    np.save(x_file, X_features)
    np.save(y_file, Y_final)

    print(f"SAVED → {x_file} : {X_features.shape}")
    print(f"SAVED → {y_file} : {Y_final.shape}")


# ---------------- RUN ----------------

if __name__ == "__main__":

    print("========== TRAIN/VAL SUBJECTS (1–30) ==========")
    for sid in TRAIN_VAL_SUBJECTS:
        try:
            process_subject(sid)
        except Exception as e:
            print(f"[ERROR] Subject {sid:02d} failed: {e}")

    print("\n========== TEST SUBJECTS (31–32) ==========")
    for sid in TEST_SUBJECTS:
        try:
            process_subject(sid, prefix="test_")
        except Exception as e:
            print(f"[ERROR] Test subject {sid:02d} failed: {e}")

    print("\n[DONE] All subjects saved to data/processed/")

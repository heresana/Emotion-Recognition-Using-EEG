#!/usr/bin/env python3
"""
02b_hyperparameter_tuning.py (FAST MODE, confusion plots removed)
-----------------------------------------------------
K-Fold hyperparameter tuning for BiLSTM+MHSA on DEAP EEG data
Binary classification (high vs low) for Valence and Arousal
Plots only accuracy/loss curves.
"""

import os, gc, csv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from tensorflow.keras.layers import Input, MultiHeadAttention, LayerNormalization, Add, Bidirectional, LSTM, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

# ==========================================================
# SEED & CONFIGURATION
# ==========================================================
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

DATA_DIR = "data/processed"
RESULTS_DIR = "results_hyper_tuning_bilstm"
MODELS_DIR = "models_hyper_tuning_bilstm"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

SUMMARY_CSV = os.path.join(RESULTS_DIR, "hyperparameter_results.csv")
BEST_CONFIG_FILE = os.path.join(RESULTS_DIR, "best_config.txt")
FINAL_MODEL_FILE = os.path.join(MODELS_DIR, "best_model.h5")

# ==========================================================
# TRAIN / VAL SUBJECTS
# ==========================================================
TRAIN_VAL_SUBJECTS = [f"s{i:02d}" for i in range(1, 31)]  # 1–30

# ==========================================================
# FAST MODE ⚡
# ==========================================================
FAST_MODE = True
if FAST_MODE:
    TRAIN_VAL_SUBJECTS = TRAIN_VAL_SUBJECTS[:4]
    KFOLDS = 5
    BATCH_SIZES = [64]
    EPOCHS_LIST = [80]
    LSTM_UNITS = [64]
    DROPOUTS = [0.3]
    LRS = [5e-4]
else:
    KFOLDS = 5
    BATCH_SIZES = [64]
    EPOCHS_LIST = [100, 120]
    LSTM_UNITS = [64]
    DROPOUTS = [0.3]
    LRS = [5e-4]

# ==========================================================
# UTILS: MAPPING & PLOTTING
# ==========================================================
def map_valence_arousal(valence, arousal, threshold=5.0):
    v_label = 1 if valence >= threshold else 0
    a_label = 1 if arousal >= threshold else 0
    return np.array([v_label, a_label], dtype=np.int32)

def plot_accuracy_loss(history, dimension, save_path):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history[f'{dimension}_accuracy'], 'g-', label='Train Acc')
    plt.plot(history.history[f'val_{dimension}_accuracy'], 'g--', label='Val Acc')
    plt.plot(history.history[f'{dimension}_loss'], 'r-', label='Train Loss')
    plt.plot(history.history[f'val_{dimension}_loss'], 'r--', label='Val Loss')
    plt.title(f"Accuracy and Loss – {dimension.upper()}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy / Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# ==========================================================
# MODEL CREATION
# ==========================================================
def create_bilstm_va(input_shape, lstm_units=64, dropout=0.3, lr=1e-4, num_heads=4):
    inp = Input(shape=input_shape)
    x = Bidirectional(LSTM(lstm_units, return_sequences=True))(inp)
    x = Dropout(dropout)(x)

    attn_out = MultiHeadAttention(num_heads=num_heads, key_dim=lstm_units)(x, x)
    attn_out = Add()([x, attn_out])
    attn_out = LayerNormalization()(attn_out)

    x = Bidirectional(LSTM(lstm_units))(attn_out)
    x = Dropout(dropout)(x)

    valence_out = Dense(1, activation='sigmoid', name='valence')(x)
    arousal_out = Dense(1, activation='sigmoid', name='arousal')(x)

    model = Model(inputs=inp, outputs=[valence_out, arousal_out])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss={'valence': 'binary_crossentropy', 'arousal': 'binary_crossentropy'},
        metrics=['accuracy']
    )
    return model

# ==========================================================
# INITIALIZE RESULTS FILE
# ==========================================================
if not os.path.exists(SUMMARY_CSV):
    with open(SUMMARY_CSV, "w", newline="") as f:
        csv.writer(f).writerow([
            "Batch", "Epochs", "LSTM", "Dropout", "LR",
            "Mean_Valence_Acc", "Mean_Arousal_Acc", "Mean_Overall_Acc"
        ])

# ==========================================================
# LOAD TRAIN/VAL SUBJECT DATA
# ==========================================================
print("📦 Loading train/val subjects...")
X_all, Yv_all, Ya_all = [], [], []

for subj in TRAIN_VAL_SUBJECTS:
    x_path = os.path.join(DATA_DIR, f"{subj}_X.npy")
    y_path = os.path.join(DATA_DIR, f"{subj}_Y.npy")
    if not (os.path.exists(x_path) and os.path.exists(y_path)):
        print(f"[WARN] Missing files for {subj}, skipping.")
        continue

    x = np.load(x_path).astype(np.float32)
    y = np.load(y_path).astype(np.float32)

    mapped = np.array([map_valence_arousal(v, a) for v, a in y])
    Yv_all.append(mapped[:, 0])
    Ya_all.append(mapped[:, 1])
    X_all.append(x)

X_all = np.concatenate(X_all, axis=0)
Yv_all = np.concatenate(Yv_all, axis=0)
Ya_all = np.concatenate(Ya_all, axis=0)
print(f"✅ Combined train/val dataset: {X_all.shape}, Labels: {Yv_all.shape}, {Ya_all.shape}")

# ==========================================================
# HYPERPARAMETER TUNING (K-FOLD)
# ==========================================================
all_results = []

for lstm_units in LSTM_UNITS:
    for dropout in DROPOUTS:
        for lr in LRS:
            for batch_size in BATCH_SIZES:
                for epochs in EPOCHS_LIST:
                    print("\n" + "=" * 60)
                    print(f"🔍 LSTM={lstm_units}, Dropout={dropout}, LR={lr}, Batch={batch_size}, Epochs={epochs}")
                    print("=" * 60)

                    kf = KFold(n_splits=KFOLDS, shuffle=True, random_state=SEED)
                    val_v_accs, val_a_accs = [], []

                    for fold, (train_idx, val_idx) in enumerate(kf.split(X_all), 1):
                        x_train, x_val = X_all[train_idx], X_all[val_idx]
                        yv_train, yv_val = Yv_all[train_idx], Yv_all[val_idx]
                        ya_train, ya_val = Ya_all[train_idx], Ya_all[val_idx]

                        model = create_bilstm_va((X_all.shape[1], X_all.shape[2]),
                                                 lstm_units=lstm_units, dropout=dropout, lr=lr)

                        callbacks = [
                            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0)
                        ]

                        history = model.fit(
                            x_train,
                            {'valence': yv_train, 'arousal': ya_train},
                            validation_data=(x_val, {'valence': yv_val, 'arousal': ya_val}),
                            epochs=epochs,
                            batch_size=batch_size,
                            verbose=0,
                            callbacks=callbacks
                        )

                        # --- last fold accuracy/loss curves ---
                        if fold == KFOLDS:
                            plot_accuracy_loss(history, 'valence', os.path.join(RESULTS_DIR, "valence_training_curve.png"))
                            plot_accuracy_loss(history, 'arousal', os.path.join(RESULTS_DIR, "arousal_training_curve.png"))
                        
                        # --- predictions ---
                        train_preds = model.predict(x_train, verbose=0)
                        train_v_pred = (train_preds[0].ravel() > 0.5).astype(int)
                        train_a_pred = (train_preds[1].ravel() > 0.5).astype(int)

                        val_preds = model.predict(x_val, verbose=0)
                        val_v_pred = (val_preds[0].ravel() > 0.5).astype(int)
                        val_a_pred = (val_preds[1].ravel() > 0.5).astype(int)

                        # --- compute accuracies ---
                        train_v_acc = accuracy_score(yv_train, train_v_pred)
                        train_a_acc = accuracy_score(ya_train, train_a_pred)
                        val_v_acc = accuracy_score(yv_val, val_v_pred)
                        val_a_acc = accuracy_score(ya_val, val_a_pred)

                        val_v_accs.append(val_v_acc)
                        val_a_accs.append(val_a_acc)

                        # --- print per fold ---
                        print(f"Fold {fold} | Train Acc -> Valence: {train_v_acc:.4f}, Arousal: {train_a_acc:.4f} | "
                            f"Val Acc -> Valence: {val_v_acc:.4f}, Arousal: {val_a_acc:.4f}")


                    mean_val_v = np.mean(val_v_accs)
                    mean_val_a = np.mean(val_a_accs)
                    mean_val_overall = (mean_val_v + mean_val_a) / 2

                    with open(SUMMARY_CSV, "a", newline="") as f:
                        csv.writer(f).writerow([
                            batch_size, epochs, lstm_units, dropout, lr,
                            f"{mean_val_v:.4f}", f"{mean_val_a:.4f}", f"{mean_val_overall:.4f}"
                        ])

                    all_results.append((batch_size, epochs, lstm_units, dropout, lr,
                                        mean_val_v, mean_val_a, mean_val_overall))

# ==========================================================
# BEST HYPERPARAMETERS
# ==========================================================
best = max(all_results, key=lambda x: x[7])  # highest mean_val_overall
best_batch, best_epochs, best_lstm, best_drop, best_lr, bmv, bma, bmo = best

print("\n BEST CONFIGURATION FOUND ")
print(f"Batch={best_batch}, Epochs={best_epochs}, LSTM={best_lstm}, Dropout={best_drop}, LR={best_lr}")
print(f"Validation Accuracies -> Valence: {bmv:.4f}, Arousal: {bma:.4f}, Overall: {bmo:.4f}")

# ==========================================================
# TRAIN FINAL MODEL ON ALL TRAIN/VAL SUBJECTS
# ==========================================================
print("\n📦 Training final model on all train/val subjects...")
final_model = create_bilstm_va((X_all.shape[1], X_all.shape[2]),
                               lstm_units=best_lstm, dropout=best_drop, lr=best_lr)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
]

history = final_model.fit(
    X_all,
    {'valence': Yv_all, 'arousal': Ya_all},
    validation_split=0.1,
    epochs=best_epochs,
    batch_size=best_batch,
    verbose=1,
    callbacks=callbacks
)

# --- Compute training accuracy ---
train_preds = final_model.predict(X_all, verbose=0)
train_v_pred = (train_preds[0].ravel() > 0.5).astype(int)
train_a_pred = (train_preds[1].ravel() > 0.5).astype(int)
train_v_acc = accuracy_score(Yv_all, train_v_pred)
train_a_acc = accuracy_score(Ya_all, train_a_pred)
train_overall_acc = (train_v_acc + train_a_acc) / 2

# --- Save final model ---
final_model.save(FINAL_MODEL_FILE)
print(f"✅ Final model saved to {FINAL_MODEL_FILE}")

# --- Save best config including training/validation accuracies ---
with open(BEST_CONFIG_FILE, "w") as f:
    f.write(
        f"Best Batch: {best_batch}\n"
        f"Best Epochs: {best_epochs}\n"
        f"LSTM Units: {best_lstm}\n"
        f"Dropout: {best_drop}\n"
        f"LR: {best_lr}\n\n"
        f"Training Accuracy -> Valence: {train_v_acc:.4f}, Arousal: {train_a_acc:.4f}, Overall: {train_overall_acc:.4f}\n"
        f"Validation Accuracy -> Valence: {bmv:.4f}, Arousal: {bma:.4f}, Overall: {bmo:.4f}\n"
    )
print("✅ Best configuration and training/validation accuracies saved to", BEST_CONFIG_FILE)

# ==========================================================
# CLEANUP
# ==========================================================
tf.keras.backend.clear_session()
gc.collect()

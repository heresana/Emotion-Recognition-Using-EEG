# #!/usr/bin/env python3
# """
# train_bilstm_kfold_overall.py
# -------------------------------------------
# K-Fold training using best hyperparameters,
# prints per-fold train/val accuracy for Valence and Arousal,
# computes overall confusion matrices, saves best model,
# plots smooth overall accuracy/loss curves, and saves overall results CSV.
# """

# import os, gc, csv
# import numpy as np
# import tensorflow as tf
# import matplotlib.pyplot as plt
# from sklearn.model_selection import KFold
# from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
# from tensorflow.keras.layers import Input, Bidirectional, LSTM, Dropout, Dense, MultiHeadAttention, LayerNormalization, Add
# from tensorflow.keras.models import Model
# from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# # ==========================================================
# # CONFIG
# # ==========================================================
# SEED = 42
# np.random.seed(SEED)
# tf.random.set_seed(SEED)

# DATA_DIR = "data/processed"
# RESULTS_DIR = "results_final_bilstm_va"
# MODELS_DIR = "models_final_bilstm_va"
# os.makedirs(RESULTS_DIR, exist_ok=True)
# os.makedirs(MODELS_DIR, exist_ok=True)

# BEST_CONFIG_FILE = os.path.join("results_hyper_tuning_bilstm", "best_config.txt")
# FINAL_MODEL_PATH = os.path.join(MODELS_DIR, "bilstm_best_va.h5")
# SUBJECTS = [f"s{i:02d}" for i in range(1, 31)]
# KFOLDS = 5

# # ==========================================================
# # LOAD BEST CONFIG
# # ==========================================================
# def load_best_config():
#     defaults = {'Batch': 64, 'Epochs': 100, 'LSTM Units': 64, 'Dropout': 0.3, 'LR': 4.5e-4}
#     if os.path.exists(BEST_CONFIG_FILE):
#         with open(BEST_CONFIG_FILE, "r") as f:
#             for line in f:
#                 parts = line.strip().split(":")
#                 if len(parts) == 2:
#                     key, val = parts
#                     key, val = key.strip(), val.strip()
#                     if key in defaults:
#                         defaults[key] = float(val) if '.' in val else int(val)
#     return (
#         int(defaults['Batch']),
#         int(defaults['Epochs']),
#         int(defaults['LSTM Units']),
#         float(defaults['Dropout']),
#         float(defaults['LR'])
#     )

# batch_size, epochs, lstm_units, dropout, lr = load_best_config()
# print(f"[INFO] Using best config: batch={batch_size}, epochs={epochs}, LSTM={lstm_units}, dropout={dropout}, LR={lr}")

# # ==========================================================
# # MODEL CREATION
# # ==========================================================
# def create_bilstm_va(input_shape, lstm_units=64, dropout=0.3, lr=1e-4, num_heads=4):
#     inp = Input(shape=input_shape)
#     x = Bidirectional(LSTM(lstm_units, return_sequences=True))(inp)
#     x = Dropout(dropout)(x)

#     attn_out = MultiHeadAttention(num_heads=num_heads, key_dim=lstm_units)(x, x)
#     attn_out = Add()([x, attn_out])
#     attn_out = LayerNormalization()(attn_out)

#     x = Bidirectional(LSTM(lstm_units))(attn_out)
#     x = Dropout(dropout)(x)

#     valence_out = Dense(1, activation='sigmoid', name='valence')(x)
#     arousal_out = Dense(1, activation='sigmoid', name='arousal')(x)

#     model = Model(inputs=inp, outputs=[valence_out, arousal_out])
#     model.compile(
#         optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
#         loss={'valence': 'binary_crossentropy', 'arousal': 'binary_crossentropy'},
#         metrics=['accuracy']
#     )
#     return model

# # ==========================================================
# # LOAD DATA
# # ==========================================================
# print("📦 Loading processed features...")
# X_all, Yv_all, Ya_all = [], [], []
# for subj in SUBJECTS:
#     x_path = os.path.join(DATA_DIR, f"{subj}_X.npy")
#     y_path = os.path.join(DATA_DIR, f"{subj}_Y.npy")
#     if not (os.path.exists(x_path) and os.path.exists(y_path)):
#         continue
#     x = np.load(x_path).astype(np.float32)
#     y = np.load(y_path).astype(np.int32)
#     X_all.append(x)
#     Yv_all.append(y[:, 0])
#     Ya_all.append(y[:, 1])

# if len(X_all) == 0:
#     raise ValueError("No data found!")

# X_all = np.concatenate(X_all)
# Yv_all = np.concatenate(Yv_all)
# Ya_all = np.concatenate(Ya_all)
# print(f"✅ Data loaded: {X_all.shape[0]} samples")

# # ==========================================================
# # K-FOLD TRAINING
# # ==========================================================
# kf = KFold(n_splits=KFOLDS, shuffle=True, random_state=SEED)
# best_val_overall = 0.0

# # Store per-fold history for averaging
# history_per_fold = {'valence_loss':[], 'val_valence_loss':[], 'arousal_loss':[], 'val_arousal_loss':[],
#                     'valence_accuracy':[], 'val_valence_accuracy':[], 'arousal_accuracy':[], 'val_arousal_accuracy':[]}

# # Collect overall validation predictions for confusion matrices
# all_yv_true, all_yv_pred = [], []
# all_ya_true, all_ya_pred = [], []

# for fold, (train_idx, val_idx) in enumerate(kf.split(X_all), 1):
#     print(f"\n🧩 Fold {fold}/{KFOLDS}")
#     x_train, x_val = X_all[train_idx], X_all[val_idx]
#     yv_train, yv_val = Yv_all[train_idx], Yv_all[val_idx]
#     ya_train, ya_val = Ya_all[train_idx], Ya_all[val_idx]

#     model = create_bilstm_va((X_all.shape[1], X_all.shape[2]), lstm_units=lstm_units, dropout=dropout, lr=lr)

#     checkpoint = ModelCheckpoint(FINAL_MODEL_PATH, monitor='val_loss', save_best_only=True, verbose=1)
#     early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

#     history = model.fit(
#         x_train,
#         {'valence': yv_train, 'arousal': ya_train},
#         validation_data=(x_val, {'valence': yv_val, 'arousal': ya_val}),
#         epochs=epochs,
#         batch_size=batch_size,
#         verbose=1,
#         callbacks=[checkpoint, early_stop]
#     )

#     # Store per-fold histories for averaging
#     for key in history_per_fold.keys():
#         history_per_fold[key].append(history.history[key])

#     # Predictions
#     train_preds = model.predict(x_train, verbose=0)
#     val_preds = model.predict(x_val, verbose=0)

#     yv_train_pred = (train_preds[0].ravel() > 0.5).astype(int)
#     ya_train_pred = (train_preds[1].ravel() > 0.5).astype(int)
#     yv_val_pred = (val_preds[0].ravel() > 0.5).astype(int)
#     ya_val_pred = (val_preds[1].ravel() > 0.5).astype(int)

#     # Per-fold accuracies
#     train_v_acc = accuracy_score(yv_train, yv_train_pred)
#     train_a_acc = accuracy_score(ya_train, ya_train_pred)
#     val_v_acc = accuracy_score(yv_val, yv_val_pred)
#     val_a_acc = accuracy_score(ya_val, ya_val_pred)

#     print(f"[Fold {fold}] Train Acc: Valence={train_v_acc:.4f}, Arousal={train_a_acc:.4f}")
#     print(f"[Fold {fold}] Val Acc:   Valence={val_v_acc:.4f}, Arousal={val_a_acc:.4f}")

#     # Collect overall validation predictions
#     all_yv_true.append(yv_val); all_yv_pred.append(yv_val_pred)
#     all_ya_true.append(ya_val); all_ya_pred.append(ya_val_pred)

# # ==========================================================
# # SMOOTH OVERALL CURVES
# # ==========================================================
# overall_history = {}
# for key, fold_histories in history_per_fold.items():
#     overall_history[key] = np.mean(np.array(fold_histories), axis=0)  # average across folds

# def plot_overall_curves(history_dict, save_dir=RESULTS_DIR):
#     # Valence
#     plt.figure(figsize=(8,5))
#     plt.plot(history_dict['valence_accuracy'], 'g-', label='Train Acc')
#     plt.plot(history_dict['val_valence_accuracy'], 'k--', label='Val Acc')
#     plt.plot(history_dict['valence_loss'], 'r-', label='Train Loss')
#     plt.plot(history_dict['val_valence_loss'], 'r--', label='Val Loss')
#     plt.title("Overall Valence Accuracy/Loss")
#     plt.xlabel("Epoch")
#     plt.ylabel("Value")
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig(os.path.join(save_dir, "valence_curves_overall.png"), dpi=300)
#     plt.close()

#     # Arousal
#     plt.figure(figsize=(8,5))
#     plt.plot(history_dict['arousal_accuracy'], 'g-', label='Train Acc')
#     plt.plot(history_dict['val_arousal_accuracy'], 'k--', label='Val Acc')
#     plt.plot(history_dict['arousal_loss'], 'r-', label='Train Loss')
#     plt.plot(history_dict['val_arousal_loss'], 'r--', label='Val Loss')
#     plt.title("Overall Arousal Accuracy/Loss")
#     plt.xlabel("Epoch")
#     plt.ylabel("Value")
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig(os.path.join(save_dir, "arousal_curves_overall.png"), dpi=300)
#     plt.close()

# plot_overall_curves(overall_history)
# print("✅ Overall accuracy/loss curves saved.")

# # ==========================================================
# # COMPUTE OVERALL TRAIN & VALIDATION ACCURACIES
# # ==========================================================
# final_model = tf.keras.models.load_model(FINAL_MODEL_PATH)

# # Training overall accuracy
# train_preds = final_model.predict(X_all, verbose=0)
# overall_train_val_pred = (train_preds[0].ravel() > 0.5).astype(int)
# overall_train_ar_pred = (train_preds[1].ravel() > 0.5).astype(int)

# overall_train_val_acc = accuracy_score(Yv_all, overall_train_val_pred)
# overall_train_ar_acc = accuracy_score(Ya_all, overall_train_ar_pred)

# # Validation overall accuracy from K-Fold
# all_yv_true = np.concatenate(all_yv_true)
# all_yv_pred = np.concatenate(all_yv_pred)
# all_ya_true = np.concatenate(all_ya_true)
# all_ya_pred = np.concatenate(all_ya_pred)

# overall_val_val_acc = accuracy_score(all_yv_true, all_yv_pred)
# overall_val_ar_acc = accuracy_score(all_ya_true, all_ya_pred)

# print("\n📊 Overall Accuracies:")
# print(f"Training Acc -> Valence: {overall_train_val_acc:.4f}, Arousal: {overall_train_ar_acc:.4f}")
# print(f"Validation Acc -> Valence: {overall_val_val_acc:.4f}, Arousal: {overall_val_ar_acc:.4f}")

# # ==========================================================
# # SAVE OVERALL RESULTS TO CSV
# # ==========================================================
# csv_file = os.path.join(RESULTS_DIR, "overall_train_val_acc.csv")
# with open(csv_file, "w", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerow([
#         "overall_train_valence_acc",
#         "overall_train_arousal_acc",
#         "overall_validation_valence_acc",
#         "overall_validation_arousal_acc"
#     ])
#     writer.writerow([
#         overall_train_val_acc,
#         overall_train_ar_acc,
#         overall_val_val_acc,
#         overall_val_ar_acc
#     ])
# print(f"✅ Overall results saved to {csv_file}")

# # ==========================================================
# # OVERALL CONFUSION MATRICES
# # ==========================================================
# cm_v = confusion_matrix(all_yv_true, all_yv_pred)
# disp_v = ConfusionMatrixDisplay(cm_v, display_labels=["Low", "High"])
# disp_v.plot(cmap=plt.cm.Blues)
# plt.title("Valence")
# plt.savefig(os.path.join(RESULTS_DIR, "confusion_valence_overall.png"), dpi=300)
# plt.close()

# cm_a = confusion_matrix(all_ya_true, all_ya_pred)
# disp_a = ConfusionMatrixDisplay(cm_a, display_labels=["Low", "High"])
# disp_a.plot(cmap=plt.cm.Oranges)
# plt.title("Arousal")
# plt.savefig(os.path.join(RESULTS_DIR, "confusion_arousal_overall.png"), dpi=300)
# plt.close()
# print("✅ Overall confusion matrices saved.")

# tf.keras.backend.clear_session()
# gc.collect()
# print(f"✅ Training finished. Best model saved at {FINAL_MODEL_PATH}")


"""
train_bilstm_kfold_overall.py
-------------------------------------------
K-Fold training using best hyperparameters,
prints per-fold train/val accuracy for Valence and Arousal,
computes overall confusion matrices, saves best model,
plots smooth overall accuracy/loss curves, and saves overall results CSV.
"""

import os, gc, csv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.layers import Input, Bidirectional, LSTM, Dropout, Dense, MultiHeadAttention, LayerNormalization, Add
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# ==========================================================
# CONFIG
# ==========================================================
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

DATA_DIR = "data/processed"
RESULTS_DIR = "results_final_bilstm_va"
MODELS_DIR = "models_final_bilstm_va"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

BEST_CONFIG_FILE = os.path.join("results_hyper_tuning_bilstm", "best_config.txt")
FINAL_MODEL_PATH = os.path.join(MODELS_DIR, "bilstm_best_va.h5")
SUBJECTS = [f"s{i:02d}" for i in range(1, 31)]
KFOLDS = 5

# ==========================================================
# LOAD BEST CONFIG
# ==========================================================
def load_best_config():
    defaults = {'Batch': 64, 'Epochs': 100, 'LSTM Units': 64, 'Dropout': 0.3, 'LR': 4.5e-4}
    if os.path.exists(BEST_CONFIG_FILE):
        with open(BEST_CONFIG_FILE, "r") as f:
            for line in f:
                parts = line.strip().split(":")
                if len(parts) == 2:
                    key, val = parts
                    key, val = key.strip(), val.strip()
                    if key in defaults:
                        defaults[key] = float(val) if '.' in val else int(val)
    return (
        int(defaults['Batch']),
        int(defaults['Epochs']),
        int(defaults['LSTM Units']),
        float(defaults['Dropout']),
        float(defaults['LR'])
    )

batch_size, epochs, lstm_units, dropout, lr = load_best_config()
print(f"[INFO] Using best config: batch={batch_size}, epochs={epochs}, LSTM={lstm_units}, dropout={dropout}, LR={lr}")


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
# LOAD DATA
# ==========================================================
print("📦 Loading processed features...")
X_all, Yv_all, Ya_all = [], [], []

for subj in SUBJECTS:
    x_path = os.path.join(DATA_DIR, f"{subj}_X.npy")
    y_path = os.path.join(DATA_DIR, f"{subj}_Y.npy")

    if not (os.path.exists(x_path) and os.path.exists(y_path)):
        continue

    x = np.load(x_path).astype(np.float32)
    y = np.load(y_path).astype(np.int32)

    X_all.append(x)
    Yv_all.append(y[:, 0])   # valence
    Ya_all.append(y[:, 1])   # arousal
    # y[:, 2] = quadrant (ignored in this baseline)

X_all = np.concatenate(X_all)
Yv_all = np.concatenate(Yv_all).reshape(-1)
Ya_all = np.concatenate(Ya_all).reshape(-1)

print(f"✅ Data loaded: {X_all.shape[0]} samples")


# ==========================================================
# K-FOLD TRAINING
# ==========================================================
kf = KFold(n_splits=KFOLDS, shuffle=True, random_state=SEED)
best_val_overall = 0.0

history_per_fold = {
    'valence_loss':[], 'val_valence_loss':[],
    'arousal_loss':[], 'val_arousal_loss':[],
    'valence_accuracy':[], 'val_valence_accuracy':[],
    'arousal_accuracy':[], 'val_arousal_accuracy':[]
}

all_yv_true, all_yv_pred = [], []
all_ya_true, all_ya_pred = [], []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_all), 1):
    print(f"\n🧩 Fold {fold}/{KFOLDS}")

    x_train, x_val = X_all[train_idx], X_all[val_idx]
    yv_train, yv_val = Yv_all[train_idx], Yv_all[val_idx]
    ya_train, ya_val = Ya_all[train_idx], Ya_all[val_idx]

    model = create_bilstm_va((X_all.shape[1], X_all.shape[2]), lstm_units=lstm_units,
                             dropout=dropout, lr=lr)

    checkpoint = ModelCheckpoint(FINAL_MODEL_PATH, monitor='val_loss', save_best_only=True, verbose=1)
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

    history = model.fit(
        x_train,
        {'valence': yv_train, 'arousal': ya_train},
        validation_data=(x_val, {'valence': yv_val, 'arousal': ya_val}),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        callbacks=[checkpoint, early_stop]
    )

    # Save histories
    for k in history_per_fold:
        history_per_fold[k].append(history.history[k])

    # Predictions
    train_preds = model.predict(x_train, verbose=0)
    val_preds = model.predict(x_val, verbose=0)

    yv_train_pred = (train_preds[0].ravel() > 0.5).astype(int)
    ya_train_pred = (train_preds[1].ravel() > 0.5).astype(int)
    yv_val_pred = (val_preds[0].ravel() > 0.5).astype(int)
    ya_val_pred = (val_preds[1].ravel() > 0.5).astype(int)

    # Fold accuracy
    print(f"[Fold {fold}] Train Acc: Val={accuracy_score(yv_train, yv_train_pred):.4f}, "
          f"Ar={accuracy_score(ya_train, ya_train_pred):.4f}")
    print(f"[Fold {fold}] Val Acc:   Val={accuracy_score(yv_val, yv_val_pred):.4f}, "
          f"Ar={accuracy_score(ya_val, ya_val_pred):.4f}")

    all_yv_true.append(yv_val)
    all_yv_pred.append(yv_val_pred)
    all_ya_true.append(ya_val)
    all_ya_pred.append(ya_val_pred)
# ==========================================================
# SMOOTH OVERALL CURVES (fixed)
# ==========================================================
overall_history = {}
for key, fold_histories in history_per_fold.items():
    # Find the minimum number of epochs among folds to align
    min_len = min(len(h) for h in fold_histories)
    # Trim each fold history to min_len
    trimmed = [h[:min_len] for h in fold_histories]
    # Average across folds
    overall_history[key] = np.mean(np.array(trimmed), axis=0)

def plot_overall_curves(history_dict, save_dir=RESULTS_DIR):
    # Valence
    plt.figure(figsize=(10,6))
    plt.plot(history_dict['valence_accuracy'], 'g--', alpha=0.6, label='Train Acc')
    plt.plot(history_dict['val_valence_accuracy'], 'g-', label='Val Acc')
    plt.plot(history_dict['valence_loss'], 'o--', label='Train Loss')
    plt.plot(history_dict['val_valence_loss'], 'r-', label='Val Loss')
    plt.title("Valence")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy/loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "valence_curves_overall.png"), dpi=300)
    plt.close()

    # Arousal
    plt.figure(figsize=(10,6))
    plt.plot(history_dict['arousal_accuracy'], 'g--', alpha=0.6, label='Train Acc')
    plt.plot(history_dict['val_arousal_accuracy'], 'g-', label='Val Acc')
    plt.plot(history_dict['arousal_loss'], 'o--', label='Train Loss')
    plt.plot(history_dict['val_arousal_loss'], 'r-', label='Val Loss')
    plt.title("Arousal")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy/loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "arousal_curves_overall.png"), dpi=300)
    plt.close()

plot_overall_curves(overall_history)
print("✅ Overall accuracy/loss curves saved.")

# ==========================================================
# COMPUTE OVERALL TRAIN & VALIDATION ACCURACIES
# ==========================================================
final_model = tf.keras.models.load_model(FINAL_MODEL_PATH)

# Training overall accuracy
train_preds = final_model.predict(X_all, verbose=0)
overall_train_val_pred = (train_preds[0].ravel() > 0.5).astype(int)
overall_train_ar_pred = (train_preds[1].ravel() > 0.5).astype(int)

overall_train_val_acc = accuracy_score(Yv_all, overall_train_val_pred)
overall_train_ar_acc = accuracy_score(Ya_all, overall_train_ar_pred)

# Validation overall accuracy from K-Fold
all_yv_true = np.concatenate(all_yv_true)
all_yv_pred = np.concatenate(all_yv_pred)
all_ya_true = np.concatenate(all_ya_true)
all_ya_pred = np.concatenate(all_ya_pred)

overall_val_val_acc = accuracy_score(all_yv_true, all_yv_pred)
overall_val_ar_acc = accuracy_score(all_ya_true, all_ya_pred)

print("\n📊 Overall Accuracies:")
print(f"Training Acc -> Valence: {overall_train_val_acc:.4f}, Arousal: {overall_train_ar_acc:.4f}")
print(f"Validation Acc -> Valence: {overall_val_val_acc:.4f}, Arousal: {overall_val_ar_acc:.4f}")

# ==========================================================
# SAVE OVERALL RESULTS TO CSV
# ==========================================================
csv_file = os.path.join(RESULTS_DIR, "overall_train_val_acc.csv")
with open(csv_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "overall_train_valence_acc",
        "overall_train_arousal_acc",
        "overall_validation_valence_acc",
        "overall_validation_arousal_acc"
    ])
    writer.writerow([
        overall_train_val_acc,
        overall_train_ar_acc,
        overall_val_val_acc,
        overall_val_ar_acc
    ])
print(f"✅ Overall results saved to {csv_file}")

# ==========================================================
# OVERALL CONFUSION MATRICES
# ==========================================================
cm_v = confusion_matrix(all_yv_true, all_yv_pred)
disp_v = ConfusionMatrixDisplay(cm_v, display_labels=["Low", "High"])
disp_v.plot(cmap=plt.cm.Blues)
plt.title("Valence")
plt.savefig(os.path.join(RESULTS_DIR, "confusion_valence_overall.png"), dpi=300)
plt.close()

cm_a = confusion_matrix(all_ya_true, all_ya_pred)
disp_a = ConfusionMatrixDisplay(cm_a, display_labels=["Low", "High"])
disp_a.plot(cmap=plt.cm.Oranges)
plt.title("Arousal")
plt.savefig(os.path.join(RESULTS_DIR, "confusion_arousal_overall.png"), dpi=300)
plt.close()
print("✅ Overall confusion matrices saved.")

tf.keras.backend.clear_session()
gc.collect()
print(f"✅ Training finished. Best model saved at {FINAL_MODEL_PATH}")

#!/usr/bin/env python3
"""
03_train_mvgt.py  -- MVGT + BiLSTM training (valence-fix edition)

Key improvements over previous versions:
 - Adds Frontal Asymmetry (FA) feature computed from alpha band-power and appends
   it to per-channel features (helps valence learning).
 - Computes class weights and uses per-sample weights to avoid valence collapse.
 - Warm-up stage: train valence head first (with backbone frozen) so valence head
   learns before arousal/quadrant dominate gradients.
 - Unfreeze backbone and fine-tune both heads with combined sample weights.
 - Per-fold logging: accuracy & F1 for valence and arousal, CSV outputs.
 - Saves best model by mean(valence_val_acc, arousal_val_acc).
"""

import os,sys
import gc
import csv
import shutil
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Input, Dense, Dropout, Bidirectional, LSTM, GlobalAveragePooling1D, Concatenate
from tensorflow.keras.models import Model, load_model

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# custom layers (must be available in your repo)
from models.custom_layers import SpatialEncoding, GraphMultiHeadAttention

# ----------------------------
# Config
# ----------------------------
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

DATA_DIR = "data/processed"      # expects s01_X.npy, s01_Y.npy ... for subjects 1..30
RESULTS_DIR = "results_mvgt_fix_valence"
MODELS_DIR = "models_mvgt_fix_valence"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

PER_FOLD_CSV = os.path.join(RESULTS_DIR, "per_fold_results.csv")
FINAL_SUMMARY_CSV = os.path.join(RESULTS_DIR, "final_summary.csv")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_mvgt_valence_fixed.h5")

KFOLDS = 5
SUBJECTS = [f"s{i:02d}" for i in range(1, 31)]  # training/validation subjects (1..30)

# data / features
N_CH = 32
ORIG_F_FEATS = 17  # original features per channel from your extractor

# band-power assumptions
BP_START_IDX = 4  # start index of band power within the 17 features (theta, alpha, beta, gamma)
ALPHA_REL_IDX = 1  # within [theta,alpha,beta,gamma], alpha index

# model training params
D_MODEL = 64
NUM_HEADS = 2
LSTM_UNITS = 64
DROPOUT = 0.3
BATCH_SIZE = 64
TOTAL_EPOCHS = 110
FREEZE_WARMUP_EPOCHS = 15   # warm-up valence-only epochs
LR = 8e-4
CLIPNORM = 1.0
LABEL_SMOOTH = 0.03
PATIENCE = 12

# optional pretrained bilstm path (if you have older trained BiLSTM)
PRETRAINED_BILSTM_PATH = os.path.join("models_final_bilstm_va", "best_bilstm_va.h5")

# ----------------------------
# Helper: create MVGT-BiLSTM model
# (compact but functionally equivalent to your architecture)
# ----------------------------
def create_mvgt_bilstm_va(input_shape, d_model=D_MODEL, lstm_units=LSTM_UNITS, dropout_rate=DROPOUT,
                          lr=LR, num_heads=NUM_HEADS, sigma=0.02, pretrained_bilstm_weights=None,
                          freeze_bilstm_initial=False):
    """
    Returns compiled Keras model with two sigmoid outputs: valence and arousal.
    input_shape: (n_channels, n_features_per_channel)
    """
    inp = Input(shape=input_shape, name="input")

    bilstm1 = Bidirectional(LSTM(lstm_units, return_sequences=True, name='bilstm_spatial'))
    x_lstm = bilstm1(inp)
    x = Dropout(dropout_rate)(x_lstm)

    # optional pretrained bilstm weights
    if pretrained_bilstm_weights is not None:
        try:
            bilstm1.set_weights(pretrained_bilstm_weights)
            bilstm1.trainable = not freeze_bilstm_initial
            print("✅ Pretrained BiLSTM weights loaded; trainable =", bilstm1.trainable)
        except Exception as e:
            print("⚠️ Could not set pretrained BiLSTM weights:", e)

    x = Dense(d_model, name="proj")(x)

    # spatial encoding and graph attention
    x, bias = SpatialEncoding(d_model, K=N_CH, sigma=sigma)(x)

    att = GraphMultiHeadAttention(num_heads=num_heads, key_dim=d_model, dropout_rate=dropout_rate)(x, x, bias)
    x = tf.keras.layers.Add()([x, att])
    x = tf.keras.layers.LayerNormalization()(x)

    ffn = Dense(d_model * 4, activation='gelu')(x)
    ffn = Dense(d_model)(ffn)
    x = tf.keras.layers.Add()([x, ffn])
    x = tf.keras.layers.LayerNormalization()(x)

    bilstm2 = Bidirectional(LSTM(lstm_units, name='bilstm_fusion'))
    fusion_seq = bilstm2(x)

    gap = GlobalAveragePooling1D()(x_lstm)
    total = Concatenate()([fusion_seq, gap])

    total = Dense(128, activation='relu')(total)
    total = Dropout(dropout_rate)(total)

    v_out = Dense(1, activation='sigmoid', name='valence')(total)
    a_out = Dense(1, activation='sigmoid', name='arousal')(total)

    model = Model(inputs=inp, outputs=[v_out, a_out])

    opt = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=CLIPNORM)
    loss_fn = tf.keras.losses.BinaryCrossentropy(label_smoothing=LABEL_SMOOTH)

    model.compile(optimizer=opt,
                  loss={'valence': loss_fn, 'arousal': loss_fn},
                  metrics={'valence': 'accuracy', 'arousal': 'accuracy'})

    return model

# ----------------------------
# Load processed features for all subjects and append FA feature
# ----------------------------
print("📦 Loading processed features from", DATA_DIR)
X_list, Yv_list, Ya_list = [], [], []

for subj in SUBJECTS:
    x_path = os.path.join(DATA_DIR, f"{subj}_X.npy")
    y_path = os.path.join(DATA_DIR, f"{subj}_Y.npy")
    if not (os.path.exists(x_path) and os.path.exists(y_path)):
        print(f"Warning: missing {subj} files; skipping subject.")
        continue
    x = np.load(x_path).astype(np.float32)  # expected (n_epochs, 32, 17)
    y = np.load(y_path, allow_pickle=True)
    y = np.asarray(y)

    # coerce shape (some saved were (n,17,32))
    if x.ndim == 3 and x.shape[1] == ORIG_F_FEATS and x.shape[2] == N_CH:
        x = np.swapaxes(x, 1, 2)

    if x.ndim != 3 or x.shape[1] != N_CH or x.shape[2] != ORIG_F_FEATS:
        raise RuntimeError(f"Unexpected shape for {x_path}: {x.shape} (expected (n,32,17))")

    # Y layout: might be (n,3) or (n,2) or (n,)
    # ensure we have arrays for valence and arousal
    if y.ndim == 1:
        # fallback: scalar quadrants -> use decode? but this is rare in processed stage
        raise RuntimeError(f"Y for {subj} is 1D (no V/A columns). Provide Y arrays with valence/arousal.")
    Yv = np.asarray(y[:, 0], dtype=np.int32)
    Ya = np.asarray(y[:, 1], dtype=np.int32)

    X_list.append(x)
    Yv_list.append(Yv)
    Ya_list.append(Ya)

if len(X_list) == 0:
    raise RuntimeError("No processed data found in DATA_DIR. Check paths.")

X_all = np.concatenate(X_list, axis=0)   # (N_epochs_total, 32, 17)
Yv_all = np.concatenate(Yv_list, axis=0).reshape(-1)
Ya_all = np.concatenate(Ya_list, axis=0).reshape(-1)

print("Loaded epochs:", X_all.shape[0], "shape:", X_all.shape)

# compute FA feature (alpha power difference F4 - F3) and append as extra feature
# ensure channel ordering corresponds to CHANNEL_NAMES_32 assumed earlier (DEAP Geneva ordering)
CHANNEL_NAMES_32 = [
    "Fp1","AF3","F3","F7","FC5","FC1","C3","T7",
    "CP5","CP1","P3","P7","PO3","O1","Oz","Pz",
    "Fp2","AF4","Fz","F4","F8","FC6","FC2","Cz",
    "C4","T8","CP6","CP2","P4","P8","PO4","O2"
]
idx_F3 = CHANNEL_NAMES_32.index("F3")  # 2
idx_F4 = CHANNEL_NAMES_32.index("F4")  # 19

alpha_idx = BP_START_IDX + ALPHA_REL_IDX
if alpha_idx >= ORIG_F_FEATS:
    raise RuntimeError("Alpha index configuration incorrect (alpha_idx >= ORIG_F_FEATS).")

alpha_F3 = X_all[:, idx_F3, alpha_idx]  # (N_epochs,)
alpha_F4 = X_all[:, idx_F4, alpha_idx]
FA_vals = alpha_F4 - alpha_F3  # log-power difference approximates frontal asymmetry (right - left)

# optional clipping of extremes
low_clip = np.percentile(FA_vals, 1)
high_clip = np.percentile(FA_vals, 99)
FA_vals = np.clip(FA_vals, low_clip, high_clip).astype(np.float32)

# broadcast as extra feature across channels (simple and effective)
FA_broadcast = np.tile(FA_vals[:, np.newaxis, np.newaxis], (1, N_CH, 1))
X_all_aug = np.concatenate([X_all, FA_broadcast], axis=-1)  # (N_epochs, 32, ORIG_F_FEATS+1)
F_FEATS = ORIG_F_FEATS + 1

print("Appended FA feature -> new per-channel features:", F_FEATS, "new X shape:", X_all_aug.shape)

del X_all, FA_broadcast
gc.collect()

# ----------------------------
# compute class weights and per-sample weights
# ----------------------------
unique_v, counts_v = np.unique(Yv_all, return_counts=True)
unique_a, counts_a = np.unique(Ya_all, return_counts=True)
print("Valence label counts:", dict(zip(unique_v.tolist(), counts_v.tolist())))
print("Arousal label counts:", dict(zip(unique_a.tolist(), counts_a.tolist())))

# compute balanced class weights for binary labels (0,1)
cw_v_arr = compute_class_weight('balanced', classes=np.array([0,1]), y=Yv_all)
cw_a_arr = compute_class_weight('balanced', classes=np.array([0,1]), y=Ya_all)
cw_v = {0: float(cw_v_arr[0]), 1: float(cw_v_arr[1])}
cw_a = {0: float(cw_a_arr[0]), 1: float(cw_a_arr[1])}
print("Class weights valence:", cw_v)
print("Class weights arousal:", cw_a)

# per-sample weights (for training)
sample_weights_v = np.array([cw_v[int(y)] for y in Yv_all], dtype=np.float32)
sample_weights_a = np.array([cw_a[int(y)] for y in Ya_all], dtype=np.float32)
# combine (average) to get balanced emphasis on both heads during fine-tune
sample_weights_combined = (sample_weights_v + sample_weights_a) / 2.0

# ----------------------------
# Try load pretrained BiLSTM weights (optional)
# ----------------------------
pretrained_bilstm_weights = None
if os.path.exists(PRETRAINED_BILSTM_PATH):
    try:
        pre_m = load_model(PRETRAINED_BILSTM_PATH)
        for layer in pre_m.layers:
            if isinstance(layer, tf.keras.layers.Bidirectional):
                pretrained_bilstm_weights = layer.get_weights()
                print("Loaded pretrained BiLSTM weights from", PRETRAINED_BILSTM_PATH)
                break
    except Exception as e:
        print("Could not load pretrained BiLSTM:", e)

# ----------------------------
# K-Fold training
# ----------------------------
kf = KFold(n_splits=KFOLDS, shuffle=True, random_state=SEED)

# prepare CSV headers
if os.path.exists(PER_FOLD_CSV):
    os.remove(PER_FOLD_CSV)
with open(PER_FOLD_CSV, "w", newline="") as f:
    csv.writer(f).writerow(["fold", "train_val_acc", "train_aro_acc", "val_val_acc", "val_aro_acc", "mean_val", "f1_val", "f1_aro"])

summary_rows = []
best_mean = -1.0
fold_idx = 0

for train_idx, val_idx in kf.split(X_all_aug):
    fold_idx += 1
    print(f"\n===== Fold {fold_idx}/{KFOLDS} =====")

    x_train = X_all_aug[train_idx]; x_val = X_all_aug[val_idx]
    yv_train = Yv_all[train_idx]; yv_val = Yv_all[val_idx]
    ya_train = Ya_all[train_idx]; ya_val = Ya_all[val_idx]

    sw_v_train = sample_weights_v[train_idx]
    sw_a_train = sample_weights_a[train_idx]
    sw_combined_train = sample_weights_combined[train_idx]

    # create model (freeze bilstm initially for warm-up)
    model = create_mvgt_bilstm_va((N_CH, F_FEATS),
                                  d_model=D_MODEL,
                                  lstm_units=LSTM_UNITS,
                                  dropout_rate=DROPOUT,
                                  lr=LR,
                                  num_heads=NUM_HEADS,
                                  sigma=0.02,
                                  pretrained_bilstm_weights=pretrained_bilstm_weights,
                                  freeze_bilstm_initial=True)

    fold_ckpt = os.path.join(MODELS_DIR, f"fold{fold_idx}_best.h5")
    chkpt = ModelCheckpoint(fold_ckpt, monitor='val_loss', save_best_only=True, verbose=1)
    early = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6, min_lr=1e-6, verbose=1)

    # ----------------------------
    # Warm-up: train valence head only
    # freeze backbone layers and arousal head
    # ----------------------------
    print("Stage 1 (Warm-up): Training VALENCE head only (backbone frozen)")

    # freeze all layers except those whose name contains 'valence' or the final dense layers
    for layer in model.layers:
        if ('valence' in layer.name) or layer.name.startswith("dense") or layer.name.startswith("proj"):
            layer.trainable = True
        else:
            layer.trainable = False

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR, clipnorm=CLIPNORM),
                  loss={'valence': tf.keras.losses.BinaryCrossentropy(label_smoothing=LABEL_SMOOTH),
                        'arousal': tf.keras.losses.BinaryCrossentropy(label_smoothing=LABEL_SMOOTH)},
                  metrics={'valence': 'accuracy', 'arousal': 'accuracy'})

    # We pass arousal dummy zeros because Keras requires target for each output.
    # sample_weight here emphasizes valence learning (applied to overall loss but arousal target is zeroed).
    warm_epochs = min(FREEZE_WARMUP_EPOCHS, max(3, TOTAL_EPOCHS // 8))
    history_warm = model.fit(
        x_train,
        {'valence': yv_train, 'arousal': np.zeros_like(ya_train)},
        validation_data=(x_val, {'valence': yv_val, 'arousal': np.zeros_like(ya_val)}),
        epochs=warm_epochs,
        batch_size=BATCH_SIZE,
        verbose=1,
        callbacks=[chkpt, early, reduce_lr],
        sample_weight=sw_v_train
    )

    # ----------------------------
    # Unfreeze and fine-tune both heads with combined sample weights
    # ----------------------------
    print("Stage 2: Unfreeze all layers and fine-tune VALENCE + AROUSAL")

    for layer in model.layers:
        layer.trainable = True

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR, clipnorm=CLIPNORM),
                  loss={'valence': tf.keras.losses.BinaryCrossentropy(label_smoothing=LABEL_SMOOTH),
                        'arousal': tf.keras.losses.BinaryCrossentropy(label_smoothing=LABEL_SMOOTH)},
                  metrics={'valence': 'accuracy', 'arousal': 'accuracy'})

    history_ft = model.fit(
        x_train, {'valence': yv_train, 'arousal': ya_train},
        validation_data=(x_val, {'valence': yv_val, 'arousal': ya_val}),
        epochs=TOTAL_EPOCHS - warm_epochs,
        batch_size=BATCH_SIZE,
        verbose=1,
        callbacks=[chkpt, early, reduce_lr],
        sample_weight=sw_combined_train
    )

    # evaluate fold with best checkpoint (if saved)
    if os.path.exists(fold_ckpt):
        try:
            model = load_model(fold_ckpt, custom_objects={'SpatialEncoding': SpatialEncoding, 'GraphMultiHeadAttention': GraphMultiHeadAttention})
        except Exception as e:
            print("Warning: failed to reload fold checkpoint:", e)

    preds_val = model.predict(x_val, verbose=0)
    if isinstance(preds_val, (list, tuple)) and len(preds_val) == 2:
        yv_pred = (preds_val[0].ravel() > 0.5).astype(int)
        ya_pred = (preds_val[1].ravel() > 0.5).astype(int)
    else:
        arr = np.asarray(preds_val)
        if arr.ndim == 2 and arr.shape[-1] == 2:
            yv_pred = (arr[:, 0] > 0.5).astype(int)
            ya_pred = (arr[:, 1] > 0.5).astype(int)
        else:
            raise RuntimeError("Unexpected model output shape at fold evaluation.")

    acc_v = accuracy_score(yv_val, yv_pred)
    acc_a = accuracy_score(ya_val, ya_pred)
    f1_v = f1_score(yv_val, yv_pred, zero_division=0)
    f1_a = f1_score(ya_val, ya_pred, zero_division=0)
    mean_val = (acc_v + acc_a) / 2.0

    # training set evaluation
    preds_train = model.predict(x_train, verbose=0)
    if isinstance(preds_train, (list, tuple)) and len(preds_train) == 2:
        yv_pred_tr = (preds_train[0].ravel() > 0.5).astype(int)
        ya_pred_tr = (preds_train[1].ravel() > 0.5).astype(int)
    else:
        arrt = np.asarray(preds_train)
        if arrt.ndim == 2 and arrt.shape[-1] == 2:
            yv_pred_tr = (arrt[:, 0] > 0.5).astype(int)
            ya_pred_tr = (arrt[:, 1] > 0.5).astype(int)
        else:
            yv_pred_tr = np.zeros_like(yv_train)
            ya_pred_tr = np.zeros_like(ya_train)

    acc_v_tr = accuracy_score(yv_train, yv_pred_tr)
    acc_a_tr = accuracy_score(ya_train, ya_pred_tr)

    print(f"[Fold {fold_idx}] Train Acc -> Val:{acc_v_tr:.4f}  Aro:{acc_a_tr:.4f}")
    print(f"[Fold {fold_idx}] Val   Acc -> Val:{acc_v:.4f}  Aro:{acc_a:.4f}  Mean:{mean_val:.4f}")
    print(f"[Fold {fold_idx}] Val F1: {f1_v:.4f}  Aro F1: {f1_a:.4f}")

    # save per-fold CSV row
    with open(PER_FOLD_CSV, "a", newline="") as f:
        csv.writer(f).writerow([fold_idx, f"{acc_v_tr:.4f}", f"{acc_a_tr:.4f}", f"{acc_v:.4f}", f"{acc_a:.4f}", f"{mean_val:.4f}", f"{f1_v:.4f}", f"{f1_a:.4f}"])

    summary_rows.append([acc_v_tr, acc_a_tr, acc_v, acc_a, mean_val, f1_v, f1_a])

    # update best model
    if mean_val > best_mean and os.path.exists(fold_ckpt):
        best_mean = mean_val
        shutil.copy(fold_ckpt, BEST_MODEL_PATH)
        print("🔥 New best model saved to", BEST_MODEL_PATH)

    tf.keras.backend.clear_session()
    gc.collect()

# ----------------------------
# After folds: summary
# ----------------------------
if len(summary_rows) > 0:
    arr = np.array(summary_rows)
    mean_train_v = arr[:, 0].mean()
    mean_train_a = arr[:, 1].mean()
    mean_val_v = arr[:, 2].mean()
    mean_val_a = arr[:, 3].mean()
    mean_mean = arr[:, 4].mean()
    mean_f1_v = arr[:, 5].mean()
    mean_f1_a = arr[:, 6].mean()

    with open(FINAL_SUMMARY_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "valence", "arousal"])
        writer.writerow(["mean_train_acc", f"{mean_train_v:.4f}", f"{mean_train_a:.4f}"])
        writer.writerow(["mean_val_acc", f"{mean_val_v:.4f}", f"{mean_val_a:.4f}"])
        writer.writerow(["mean_mean_acc", f"{mean_mean:.4f}"])
        writer.writerow(["mean_valence_f1", f"{mean_f1_v:.4f}"])
        writer.writerow(["mean_arousal_f1", f"{mean_f1_a:.4f}"])

    print("\n===== FINAL SUMMARY =====")
    print(f"Mean Train Valence Acc : {mean_train_v:.4f}")
    print(f"Mean Train Arousal Acc : {mean_train_a:.4f}")
    print(f"Mean Valence Val Acc  : {mean_val_v:.4f}")
    print(f"Mean Arousal Val Acc  : {mean_val_a:.4f}")
    print(f"Mean Mean Acc         : {mean_mean:.4f}")
    print(f"Mean Valence F1       : {mean_f1_v:.4f}")
    print(f"Mean Arousal F1       : {mean_f1_a:.4f}")

# ----------------------------
# Final: evaluate best model on full dataset and save confusion matrices
# ----------------------------
if os.path.exists(BEST_MODEL_PATH):
    try:
        final_model = load_model(BEST_MODEL_PATH, custom_objects={'SpatialEncoding': SpatialEncoding, 'GraphMultiHeadAttention': GraphMultiHeadAttention})
        preds_all = final_model.predict(X_all_aug, verbose=0)
        if isinstance(preds_all, (list, tuple)) and len(preds_all) == 2:
            yv_all_pred = (preds_all[0].ravel() > 0.5).astype(int)
            ya_all_pred = (preds_all[1].ravel() > 0.5).astype(int)

            # confusion matrices
            cm_v = confusion_matrix(Yv_all, yv_all_pred)
            cm_a = confusion_matrix(Ya_all, ya_all_pred)

            fig, ax = plt.subplots(figsize=(5,4))
            ConfusionMatrixDisplay(cm_v, display_labels=["Low","High"]).plot(ax=ax, cmap=plt.cm.Blues, values_format='d')
            plt.title("Valence (full eval)")
            plt.savefig(os.path.join(RESULTS_DIR, "confusion_valence_full.png"), dpi=300, bbox_inches='tight')
            plt.close()

            fig, ax = plt.subplots(figsize=(5,4))
            ConfusionMatrixDisplay(cm_a, display_labels=["Low","High"]).plot(ax=ax, cmap=plt.cm.Oranges, values_format='d')
            plt.title("Arousal (full eval)")
            plt.savefig(os.path.join(RESULTS_DIR, "confusion_arousal_full.png"), dpi=300, bbox_inches='tight')
            plt.close()

            print("✅ Final confusion matrices saved in results.")
    except Exception as e:
        print("Warning: could not evaluate final model on full set:", e)
else:
    print("⚠️ Best model not found; skip final full-evaluation.")

print("✅ Training script finished. Best model path (if any):", BEST_MODEL_PATH)

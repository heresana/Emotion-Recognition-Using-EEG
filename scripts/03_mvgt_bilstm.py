# ============================================================
# 03_train_mvgt_bilstm_va_freeze_then_unfreeze_full.py
# FINAL FULL VERSION (Quadrant-safe, unchanged architecture)
# with final confusion matrices (Option B)
# ============================================================

import os, gc, csv, shutil
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.layers import Input, Bidirectional, LSTM, Dropout, Dense, LayerNormalization, Add, Embedding, GlobalAveragePooling1D, Concatenate
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

# ----------------------------
# CONFIG
# ----------------------------
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

DATA_DIR = "data/processed"
RESULTS_DIR = "1results_freeze_unfreeze_mvgt_bilstm"
MODELS_DIR = "1models_freeze_unfreeze_mvgt_bilstm"
PER_FOLD_CSV = os.path.join(RESULTS_DIR, "per_fold_results.csv")
SUMMARY_CSV = os.path.join(RESULTS_DIR, "summary.csv")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_overall_mvgt_bilstm.h5")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)
FINAL_SUMMARY_CSV = os.path.join(RESULTS_DIR, "final_overall_summary.csv")

KFOLDS = 5
SUBJECTS = [f"s{i:02d}" for i in range(1,31)]

# MVGT / model params
N_CH = 32
F_FEATS = 17
K_GAUSSIAN = 32
D_MODEL = 64
NUM_HEADS = 2
NUM_LAYERS = 1
GAUSSIAN_SIGMA = 0.02

# training schedule
BATCH_SIZE = 64
TOTAL_EPOCHS = 110
FREEZE_EPOCHS = 20
LR = 9e-4
CLIPNORM = 1.0
LABEL_SMOOTH = 0.03
DROPOUT = 0.3
PRETRAINED_MODEL = os.path.join("models_final_bilstm_va", "best_bilstm_va.h5")

# ----------------------------
# CHANNEL POS + REGIONS
# ----------------------------
CHANNEL_POS = np.random.uniform(-1, 1, (N_CH, 3)).astype(np.float32)
CHANNEL_POS /= (np.linalg.norm(CHANNEL_POS, axis=1, keepdims=True) + 1e-6)

REGION_IDS = np.array([
    0,0,0,0,0,0,1,1,1,1,2,2,3,3,2,2,
    0,0,0,0,0,0,1,1,1,1,2,2,2,2,3,3
], dtype=np.int32)

# ----------------------------
# LOAD DATA (Quadrant-safe)
# ----------------------------
print("📦 Loading features...")
X_list, Yv_list, Ya_list = [], [], []

for subj in SUBJECTS:
    x_path = os.path.join(DATA_DIR, f"{subj}_X.npy")
    y_path = os.path.join(DATA_DIR, f"{subj}_Y.npy")
    if os.path.exists(x_path) and os.path.exists(y_path):
        x = np.load(x_path).astype(np.float32)
        y = np.load(y_path).astype(np.int32)

        # y may have shape (...,2) or (...,3)
        # we only use first two columns safely
        Yv_list.append(y[:,0])
        Ya_list.append(y[:,1])

        X_list.append(x)

if len(X_list) == 0:
    raise RuntimeError("No processed data found in DATA_DIR.")

X_all = np.concatenate(X_list, axis=0)
Yv_all = np.concatenate(Yv_list, axis=0)
Ya_all = np.concatenate(Ya_list, axis=0)

print(f"✅ Loaded total epochs: {X_all.shape[0]}, shape={X_all.shape}")

if X_all.shape[1] != N_CH:
    X_all = np.transpose(X_all, [0,2,1])
    print("⚠️ Transposed input ->", X_all.shape)

# ----------------------------
# EXTRACT PRETRAINED WEIGHTS
# ----------------------------
def extract_pretrained_bilstm_weights(path):
    if not os.path.exists(path):
        print("No pretrained model found.")
        return None
    try:
        m = load_model(path)
        for layer in m.layers:
            if isinstance(layer, tf.keras.layers.Bidirectional):
                print("✅ Using pretrained BiLSTM weights.")
                return layer.get_weights()
    except Exception as e:
        print("Error loading pretrained model:", e)
    return None

pretrained_bilstm_weights = extract_pretrained_bilstm_weights(PRETRAINED_MODEL)

# ----------------------------
# CUSTOM LAYERS
# ----------------------------
class SpatialEncoding(tf.keras.layers.Layer):
    def __init__(self, d_model, K=32, sigma=0.03):
        super().__init__()
        self.d_model = d_model
        self.K = K
        self.sigma = sigma
        self.bre_emb = Embedding(4, d_model // 4)
        self.ce_proj = Dense(d_model // 2, use_bias=False)
        self._bproj = Dense(1, use_bias=False)

    def build(self, input_shape):
        pos = tf.constant(CHANNEL_POS, dtype=tf.float32)
        diff = pos[tf.newaxis,:,:] - pos[:,tf.newaxis,:]
        dist = tf.norm(diff, axis=-1)
        min_d, max_d = tf.reduce_min(dist), tf.reduce_max(dist)
        self.dist_norm = (dist - min_d) / (max_d - min_d + 1e-6)
        self.mus = tf.linspace(0., 1., K_GAUSSIAN)

    def call(self, x):
        b = tf.shape(x)[0]
        d = self.d_model

        bre = self.bre_emb(tf.constant(REGION_IDS))[tf.newaxis,:,:]
        bre = tf.tile(bre, [b,1,1])
        if bre.shape[-1] < d:
            bre = tf.pad(bre, [[0,0],[0,0],[0,d-bre.shape[-1]]])
        x = x + bre

        dist = self.dist_norm
        mus = self.mus[tf.newaxis,tf.newaxis,:]
        B = tf.exp(-tf.square(dist[:,:,tf.newaxis]-mus)/(2*self.sigma**2))

        e = tf.reduce_sum(B, axis=1)
        ce = self.ce_proj(e)[tf.newaxis,:,:]
        ce = tf.tile(ce, [b,1,1])
        if ce.shape[-1] < d:
            ce = tf.pad(ce, [[0,0],[0,0],[0,d-ce.shape[-1]]])
        x = x + ce

        B_flat = tf.reshape(B, [self.K*self.K, K_GAUSSIAN])
        bias_flat = self._bproj(B_flat)
        bias = tf.reshape(bias_flat, [self.K, self.K])
        bias = (bias - tf.reduce_mean(bias)) / (tf.math.reduce_std(bias)+1e-6)

        return x, bias

    def get_config(self):
        return {"d_model":self.d_model,"K":self.K,"sigma":self.sigma}

class GraphMultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads, key_dim, dropout_rate=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.dropout_rate = dropout_rate
        self.head_dim = key_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q = Dense(key_dim, use_bias=False)
        self.k = Dense(key_dim, use_bias=False)
        self.v = Dense(key_dim, use_bias=False)
        self.out = Dense(key_dim)
        self.dropout = Dropout(dropout_rate)
        self.alpha = tf.Variable(0.1, trainable=True, name="alpha_adj")

    def call(self, query, value, bias=None, training=None):
        b = tf.shape(query)[0]
        n = tf.shape(query)[1]

        q, k, v = self.q(query), self.k(value), self.v(value)

        def split(x):
            return tf.transpose(tf.reshape(x,[b,n,self.num_heads,self.head_dim]),[0,2,1,3])

        qh, kh, vh = split(q), split(k), split(v)

        logits = tf.matmul(qh, kh, transpose_b=True) * self.scale
        if bias is not None:
            gb = bias[tf.newaxis,tf.newaxis,:,:]
            logits += self.alpha * gb

        attn = tf.nn.softmax(logits, axis=-1)
        attn = self.dropout(attn, training=training)

        out = tf.matmul(attn, vh)
        out = tf.transpose(out,[0,2,1,3])
        out = tf.reshape(out,[b,n,self.key_dim])
        return self.out(out)

    def get_config(self):
        return {"num_heads":self.num_heads,"key_dim":self.key_dim,"dropout_rate":self.dropout_rate}

# ----------------------------
# MODEL CREATION
# ----------------------------
def create_mvgt_bilstm_va(input_shape, lstm_units=64, dropout_rate=DROPOUT,
                          lr=LR, d_model=D_MODEL, num_heads=NUM_HEADS,
                          sigma=GAUSSIAN_SIGMA, pretrained_weights=None,
                          freeze_bilstm_initial=True):

    inp = Input(shape=input_shape)

    bilstm1 = Bidirectional(LSTM(lstm_units, return_sequences=True, name='bilstm_spatial'))
    x_lstm = bilstm1(inp)
    x = Dropout(dropout_rate)(x_lstm)

    if pretrained_weights is not None:
        try:
            bilstm1.set_weights(pretrained_weights)
            bilstm1.trainable = not freeze_bilstm_initial
            print("✅ Pretrained BiLSTM loaded (frozen).")
        except Exception as e:
            print("⚠️ Could not set pretrained weights:", e)

    x = Dense(d_model)(x)
    x, g_bias = SpatialEncoding(d_model, K=N_CH, sigma=sigma)(x)

    for _ in range(NUM_LAYERS):
        att = GraphMultiHeadAttention(num_heads, d_model, dropout_rate)(x, x, g_bias)
        x = Add()([x, att])
        x = LayerNormalization()(x)

        ffn = Dense(d_model*4, activation='gelu')(x)
        ffn = Dense(d_model)(ffn)
        x = Add()([x, ffn])
        x = LayerNormalization()(x)

    bilstm2 = Bidirectional(LSTM(64, name='bilstm_fusion'))
    fusion_seq = bilstm2(x)

    total = Concatenate()([fusion_seq, GlobalAveragePooling1D()(x_lstm)])
    total = Dense(128, activation='relu')(total)
    total = Dropout(dropout_rate)(total)

    v_out = Dense(1, activation='sigmoid', name='valence')(total)
    a_out = Dense(1, activation='sigmoid', name='arousal')(total)

    model = Model(inp, [v_out, a_out])

    opt = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=CLIPNORM)
    loss_fn = tf.keras.losses.BinaryCrossentropy(label_smoothing=LABEL_SMOOTH)

    model.compile(
        optimizer=opt,
        loss={'valence':loss_fn, 'arousal':loss_fn},
        metrics=['accuracy']
    )
    return model

# ----------------------------
# TRAINING
# ----------------------------
kf = KFold(n_splits=KFOLDS, shuffle=True, random_state=SEED)

with open(PER_FOLD_CSV, "w", newline="") as f:
    csv.writer(f).writerow(["fold","val_v_acc","val_a_acc","mean_acc","f1_v","f1_a"])

with open(FINAL_SUMMARY_CSV, "w", newline="") as f:
    csv.writer(f).writerow([
        "fold","train_valence_acc","train_arousal_acc",
        "valence_valence_acc","valence_arousal_acc",
        "overall_train_acc","overall_val_acc",
        "overall_train_f1","overall_val_f1"
    ])

summary_rows = []
best_overall = -1

# containers for final averaged plots
all_valence_train_acc = []
all_valence_train_loss = []
all_valence_val_acc = []
all_valence_val_loss = []

all_arousal_train_acc = []
all_arousal_train_loss = []
all_arousal_val_acc = []
all_arousal_val_loss = []

# ----------------------------
# NEW: accumulate overall validation true/pred across folds for final confusion matrices
# ----------------------------
all_yv_true = []
all_yv_pred = []
all_ya_true = []
all_ya_pred = []

for fold_i, (train_idx, val_idx) in enumerate(kf.split(X_all), start=1):
    print(f"\n==== Fold {fold_i}/{KFOLDS} ====")

    x_train, x_val = X_all[train_idx], X_all[val_idx]
    yv_train, yv_val = Yv_all[train_idx], Yv_all[val_idx]
    ya_train, ya_val = Ya_all[train_idx], Ya_all[val_idx]

    # class weights
    unique_v = np.unique(yv_train)
    unique_a = np.unique(ya_train)

    cw_v = compute_class_weight('balanced', classes=unique_v, y=yv_train)
    cw_a = compute_class_weight('balanced', classes=unique_a, y=ya_train)

    class_weight_v = {int(c):float(w) for c,w in zip(unique_v,cw_v)}
    class_weight_a = {int(c):float(w) for c,w in zip(unique_a,cw_a)}

    sample_weights_v = np.array([class_weight_v[int(y)] for y in yv_train])
    sample_weights_a = np.array([class_weight_a[int(y)] for y in ya_train])
    sample_weights_combined = (sample_weights_v + sample_weights_a) / 2.0

    model = create_mvgt_bilstm_va((N_CH,F_FEATS), pretrained_weights=pretrained_bilstm_weights)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6, min_lr=1e-6)
    earlystop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    fold_ckpt = os.path.join(MODELS_DIR, f"fold{fold_i}_best.h5")
    chkpt = ModelCheckpoint(fold_ckpt, monitor='val_loss', save_best_only=True)

    # Stage 1: Frozen warm-up
    initial_epochs = min(FREEZE_EPOCHS, TOTAL_EPOCHS)
    print(f"Stage 1 (frozen) training for {initial_epochs} epochs.")

    print("Training valence head...")
    history_v = model.fit(
        x_train, {"valence":yv_train, "arousal":np.zeros_like(ya_train)},
        validation_data=(x_val, {"valence":yv_val, "arousal":np.zeros_like(ya_val)}),
        epochs=initial_epochs,
        batch_size=BATCH_SIZE,
        verbose=1,
        callbacks=[reduce_lr, earlystop, chkpt],
        sample_weight=sample_weights_v
    )

    print("Training arousal head...")
    history_a = model.fit(
        x_train, {"valence":np.zeros_like(yv_train), "arousal":ya_train},
        validation_data=(x_val, {"valence":np.zeros_like(yv_val), "arousal":ya_val}),
        epochs=initial_epochs,
        batch_size=BATCH_SIZE,
        verbose=1,
        callbacks=[reduce_lr, earlystop, chkpt],
        sample_weight=sample_weights_a
    )

    # Stage 2: Unfreeze and fine-tune both
    print("Unfreezing BiLSTM for fine-tuning...")
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Bidirectional):
            layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR, clipnorm=CLIPNORM),
        loss={
            'valence':tf.keras.losses.BinaryCrossentropy(label_smoothing=LABEL_SMOOTH),
            'arousal':tf.keras.losses.BinaryCrossentropy(label_smoothing=LABEL_SMOOTH)
        },
        metrics=['accuracy']
    )

    history_ft = model.fit(
        x_train, {'valence':yv_train,'arousal':ya_train},
        validation_data=(x_val, {'valence':yv_val,'arousal':ya_val}),
        epochs=TOTAL_EPOCHS-FREEZE_EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
        callbacks=[reduce_lr,earlystop,chkpt],
        sample_weight=sample_weights_combined
    )

    # Predictions
    preds = model.predict(x_val, verbose=0)
    yv_pred = (preds[0]>0.5).astype(int).reshape(-1,)
    ya_pred = (preds[1]>0.5).astype(int).reshape(-1,)

    # ----------------------------
    # EXTEND overall arrays for final confusion matrices
    # ----------------------------
    all_yv_true.extend(yv_val.tolist())
    all_yv_pred.extend(yv_pred.tolist())
    all_ya_true.extend(ya_val.tolist())
    all_ya_pred.extend(ya_pred.tolist())

    acc_v = accuracy_score(yv_val, yv_pred)
    acc_a = accuracy_score(ya_val, ya_pred)
    f1_v = f1_score(yv_val, yv_pred, zero_division=0)
    f1_a = f1_score(ya_val, ya_pred, zero_division=0)

    mean_acc = (acc_v + acc_a) / 2
    print(f"Fold {fold_i} ACC → Valence={acc_v:.3f}, Arousal={acc_a:.3f}, Mean={mean_acc:.3f}")

    # Train set metrics
    preds_train = model.predict(x_train, verbose=0)
    yv_pred_train = (preds_train[0]>0.5).astype(int).reshape(-1,)
    ya_pred_train = (preds_train[1]>0.5).astype(int).reshape(-1,)

    acc_v_train = accuracy_score(yv_train, yv_pred_train)
    acc_a_train = accuracy_score(ya_train, ya_pred_train)
    f1_v_train = f1_score(yv_train, yv_pred_train, zero_division=0)
    f1_a_train = f1_score(ya_train, ya_pred_train, zero_division=0)

    overall_train_acc = (acc_v_train + acc_a_train) / 2
    overall_val_acc = mean_acc
    overall_train_f1 = (f1_v_train + f1_a_train) / 2
    overall_val_f1 = (f1_v + f1_a) / 2

    with open(PER_FOLD_CSV,"a",newline="") as f:
        csv.writer(f).writerow([
            fold_i,f"{acc_v:.4f}",f"{acc_a:.4f}",f"{mean_acc:.4f}",
            f"{f1_v:.4f}",f"{f1_a:.4f}"
        ])

    with open(FINAL_SUMMARY_CSV,"a",newline="") as f:
        csv.writer(f).writerow([
            fold_i,
            f"{acc_v_train:.4f}",f"{acc_a_train:.4f}",
            f"{acc_v:.4f}",f"{acc_a:.4f}",
            f"{overall_train_acc:.4f}",f"{overall_val_acc:.4f}",
            f"{overall_train_f1:.4f}",f"{overall_val_f1:.4f}"
        ])

    summary_rows.append([
        acc_v_train,acc_a_train,acc_v,acc_a,
        overall_train_acc,overall_val_acc,
        overall_train_f1,overall_val_f1
    ])

    # Collect histories
    def _find(hist, out, metric, is_val=False):
        if hist is None: return []
        keys = list(hist.history.keys())
        for k in keys:
            if is_val:
                if k.startswith("val_") and out in k and metric in k:
                    return hist.history[k]
            else:
                if not k.startswith("val_") and out in k and metric in k:
                    return hist.history[k]
        return []

    val_train_acc = _find(history_v,"valence","accuracy") + _find(history_ft,"valence","accuracy")
    val_train_loss = _find(history_v,"valence","loss") + _find(history_ft,"valence","loss")
    val_val_acc = _find(history_v,"valence","accuracy",True) + _find(history_ft,"valence","accuracy",True)
    val_val_loss = _find(history_v,"valence","loss",True) + _find(history_ft,"valence","loss",True)

    ar_train_acc = _find(history_a,"arousal","accuracy") + _find(history_ft,"arousal","accuracy")
    ar_train_loss = _find(history_a,"arousal","loss") + _find(history_ft,"arousal","loss")
    ar_val_acc = _find(history_a,"arousal","accuracy",True) + _find(history_ft,"arousal","accuracy",True)
    ar_val_loss = _find(history_a,"arousal","loss",True) + _find(history_ft,"arousal","loss",True)

    all_valence_train_acc.append(np.array(val_train_acc))
    all_valence_train_loss.append(np.array(val_train_loss))
    all_valence_val_acc.append(np.array(val_val_acc))
    all_valence_val_loss.append(np.array(val_val_loss))

    all_arousal_train_acc.append(np.array(ar_train_acc))
    all_arousal_train_loss.append(np.array(ar_train_loss))
    all_arousal_val_acc.append(np.array(ar_val_acc))
    all_arousal_val_loss.append(np.array(ar_val_loss))

    if mean_acc > best_overall:
        best_overall = mean_acc
        shutil.copy(fold_ckpt, BEST_MODEL_PATH)
        print("🔥 New best model saved.")

    tf.keras.backend.clear_session()
    gc.collect()

# ----------------------------
# After all folds summary
# ----------------------------
if len(summary_rows)>0:
    arr = np.array(summary_rows)

    mean_train_v  = arr[:,0].mean()
    mean_train_a  = arr[:,1].mean()
    mean_val_v    = arr[:,2].mean()
    mean_val_a    = arr[:,3].mean()
    mean_train    = arr[:,4].mean()
    mean_val      = arr[:,5].mean()
    mean_f1_train = arr[:,6].mean()
    mean_f1_val   = arr[:,7].mean()

    with open(FINAL_SUMMARY_CSV,"a",newline="") as f:
        csv.writer(f).writerow([
            "mean",
            f"{mean_train_v:.4f}",
            f"{mean_train_a:.4f}",
            f"{mean_val_v:.4f}",
            f"{mean_val_a:.4f}",
            f"{mean_train:.4f}",
            f"{mean_val:.4f}",
            f"{mean_f1_train:.4f}",
            f"{mean_f1_val:.4f}"
        ])

    print("\n===== FINAL OVERALL SUMMARY =====")
    print(f"Train valence acc : {mean_train_v:.4f}")
    print(f"Train arousal acc : {mean_train_a:.4f}")
    print(f"Val   valence acc : {mean_val_v:.4f}")
    print(f"Val   arousal acc : {mean_val_a:.4f}")
    print(f"Overall train acc : {mean_train:.4f}")
    print(f"Overall val   acc : {mean_val:.4f}")
    print(f"Overall train F1  : {mean_f1_train:.4f}")
    print(f"Overall val   F1  : {mean_f1_val:.4f}")

# ----------------------------
# Build averaged final plots
# ----------------------------
def _pad_stack(arrs):
    if len(arrs)==0: return None
    L = max(len(a) for a in arrs)
    M = np.full((len(arrs),L), np.nan)
    for i,a in enumerate(arrs):
        M[i,:len(a)] = a
    return M

def _mean(arrs):
    M = _pad_stack(arrs)
    if M is None: return np.array([])
    return np.nanmean(M, axis=0)

v_tr_acc = _mean(all_valence_train_acc)
v_tr_loss = _mean(all_valence_train_loss)
v_va_acc = _mean(all_valence_val_acc)
v_va_loss = _mean(all_valence_val_loss)

a_tr_acc = _mean(all_arousal_train_acc)
a_tr_loss = _mean(all_arousal_train_loss)
a_va_acc = _mean(all_arousal_val_acc)
a_va_loss = _mean(all_arousal_val_loss)

def save_plot(tr_acc, va_acc, tr_loss, va_loss, title, path):
    L = max(len(tr_acc),len(va_acc),len(tr_loss),len(va_loss))
    t = np.arange(1,L+1)

    def pad(x):
        if len(x)>=L: return x[:L]
        return np.concatenate([x, np.full(L-len(x),np.nan)])

    plt.figure(figsize=(12,9))
    plt.plot(t, pad(tr_acc), label="train_accuracy")
    plt.plot(t, pad(va_acc), label="val_accuracy")
    plt.plot(t, pad(tr_loss), '--', label="train_loss")
    plt.plot(t, pad(va_loss), '--', label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"📁 Saved: {path}")

save_plot(v_tr_acc, v_va_acc, v_tr_loss, v_va_loss,
          "Valence Accuracy & Loss (Averaged Across Folds)",
          os.path.join(PLOTS_DIR,"final_valence_overall.png"))

save_plot(a_tr_acc, a_va_acc, a_tr_loss, a_va_loss,
          "Arousal Accuracy & Loss (Averaged Across Folds)",
          os.path.join(PLOTS_DIR,"final_arousal_overall.png"))

# ----------------------------
# FINAL OVERALL CONFUSION MATRICES (Option B)
# ----------------------------
try:
    # Convert to numpy arrays
    yu_v = np.array(all_yv_true, dtype=int)
    yp_v = np.array(all_yv_pred, dtype=int)
    yu_a = np.array(all_ya_true, dtype=int)
    yp_a = np.array(all_ya_pred, dtype=int)

    if yu_v.size > 0 and yp_v.size > 0:
        cm_v = confusion_matrix(yu_v, yp_v)
        disp_v = ConfusionMatrixDisplay(cm_v, display_labels=["Low", "High"])
        fig, ax = plt.subplots(figsize=(6,5))
        disp_v.plot(ax=ax, cmap=plt.cm.Blues, values_format='d')
        plt.title("Final Valence Confusion Matrix (validation across folds)")
        out_v = os.path.join(PLOTS_DIR, "final_valence_confmat.png")
        plt.savefig(out_v, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✅ Saved final valence confusion matrix: {out_v}")
    else:
        print("⚠️ No valence validation predictions collected; skipping final confmat.")

    if yu_a.size > 0 and yp_a.size > 0:
        cm_a = confusion_matrix(yu_a, yp_a)
        disp_a = ConfusionMatrixDisplay(cm_a, display_labels=["Low", "High"])
        fig, ax = plt.subplots(figsize=(6,5))
        disp_a.plot(ax=ax, cmap=plt.cm.Oranges, values_format='d')
        plt.title("Final Arousal Confusion Matrix (validation across folds)")
        out_a = os.path.join(PLOTS_DIR, "final_arousal_confmat.png")
        plt.savefig(out_a, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✅ Saved final arousal confusion matrix: {out_a}")
    else:
        print("⚠️ No arousal validation predictions collected; skipping final confmat.")

except Exception as e:
    print("Error while generating final confusion matrices:", e)

# ----------------------------
# Copy best model
# ----------------------------
if os.path.exists(BEST_MODEL_PATH):
    final_dest = os.path.join(RESULTS_DIR,"final_overall_model.h5")
    shutil.copy(BEST_MODEL_PATH, final_dest)
    print("🎉 Best model copied to:", final_dest)
else:
    print("⚠️ Best model not found.")


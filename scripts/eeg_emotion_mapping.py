# eeg_demo_gui_mvgt.py
import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QPushButton,
    QLabel, QFileDialog, QGroupBox, QSpinBox, QCheckBox, QMessageBox
)
from PyQt6.QtCore import QTimer
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import load_model

import shap  # SHAP import (GradientExplainer used)
import mne
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib import cm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.custom_layers import SpatialEncoding, GraphMultiHeadAttention

# ----------------------------
# SHAP Visualization Popup
# ----------------------------
class ShapWindow(QWidget):
    def __init__(self, fig, parent=None):
        super().__init__(parent)
        self.setWindowTitle("SHAP Visualization")
        canvas = FigureCanvas(fig)
        layout = QVBoxLayout()
        layout.addWidget(canvas)
        self.setLayout(layout)
        self.show()

# -----------------------------
# Channel names (32-cha)
# -----------------------------
CHANNEL_NAMES_32 = [
    "Fp1","AF3","F3","F7","FC5","FC1","C3","T7",
    "CP5","CP1","P3","P7","PO3","O1","Oz","Pz",
    "Fp2","AF4","Fz","F4","F8","FC6","FC2","Cz",
    "C4","T8","CP6","CP2","P4","P8","PO4","O2"
]

# -----------------------------
# Features name(Total=17)
# -----------------------------
FEATURE_NAMES_17 = [
    "DE_theta", "DE_alpha", "DE_beta", "DE_gamma",
    "BP_theta", "BP_alpha", "BP_beta", "BP_gamma",
    "Hjorth_activity", "Hjorth_mobility", "Hjorth_complexity",
    "TD_mean", "TD_std", "TD_skewness", "TD_kurtosis", "TD_zcr", "TD_energy"
]

# -----------------------------
# Quadrant mapping (0..3)
# 0: Excited (HV-HA), 1: Tense (LV-HA), 2: Sad (LV-LA), 3: Calm (HV-LA)
# -----------------------------
QUAD_ID_TO_VA = {
    0: (1, 1),  # Excited: HV, HA
    1: (0, 1),  # Tense:   LV, HA
    2: (0, 0),  # Sad:     LV, LA
    3: (1, 0),  # Calm:    HV, LA
}
QUAD_ID_TO_NAME = {
    0: "Excited (HV-HA)",
    1: "Tense (LV-HA)",
    2: "Sad (LV-LA)",
    3: "Calm (HV-LA)",
}
VA_TO_QUAD_ID = {va: q for q, va in QUAD_ID_TO_VA.items()}

def decode_quad_y(y):
    """
    Decode GT as:
      - scalar int in [0..3] (quadrant id), or
      - one-hot/probs len 4 (argmax), or
      - [V,A,(D)] fallback -> quadrant id via VA.
    """
    try:
        arr = np.asarray(y).ravel()
        q = None
        if arr.size == 1:
            q = int(arr.item())
        elif arr.size == 4:
            q = int(np.argmax(arr))
        elif arr.size >= 2:
            v, a = int(arr[0]), int(arr[1])
            q = VA_TO_QUAD_ID.get((v, a), None)
        if q is None or q not in QUAD_ID_TO_VA:
            return f"Ground truth (Y): {arr}"
        v, a = QUAD_ID_TO_VA[q]
        return f"Quadrant GT: {q} - {QUAD_ID_TO_NAME[q]} | V={'High' if v else 'Low'} | A={'High' if a else 'Low'}"
    except Exception:
        return f"Ground truth (Y): {y}"

# -----------------------------
# Matplotlib canvas wrappers
# -----------------------------
class MultiChannelCanvas(FigureCanvas):
    def __init__(self, nchannels=32, rows=8, cols=4, figsize=(8, 9), channel_names=None):
        self.nchannels = nchannels
        self.rows = rows
        self.cols = cols
        self.channel_names = channel_names or [f"Ch{i+1}" for i in range(nchannels)]
        self.fig = Figure(figsize=figsize, tight_layout=True)
        super().__init__(self.fig)
        self.axes = []
        self.lines = []
        self._make_subplots()

    def set_channel_names(self, names):
        self.channel_names = names
        self._make_subplots()

    def _make_subplots(self):
        self.fig.clf()
        self.axes = []
        self.lines = []
        for i in range(self.nchannels):
            ax = self.fig.add_subplot(self.rows, self.cols, i + 1)
            ax.set_xticks([]); ax.set_yticks([])
            ax.axhline(0, color="#ddd", lw=0.7)
            ax.set_title(self.channel_names[i] if i < len(self.channel_names) else f"Ch{i+1}", fontsize=8)
            line, = ax.plot([], lw=0.7, color="#1f77b4")
            self.axes.append(ax)
            self.lines.append(line)
        self.draw()

    def set_header(self, text):
        self.fig.suptitle(text, fontsize=10)
        self.draw()

    def update_data(self, chunk_buffer, ylims=None):
        # chunk_buffer: (nchannels, time/feat)
        if chunk_buffer is None or chunk_buffer.size == 0:
            return
        nch = min(self.nchannels, chunk_buffer.shape[0])
        for i in range(nch):
            ch_data = chunk_buffer[i]
            self.lines[i].set_data(np.arange(ch_data.shape[0]), ch_data)
            ax = self.axes[i]
            ax.set_xlim(0, ch_data.shape[0])
            if ylims is not None:
                ax.set_ylim(ylims[0], ylims[1])
            else:
                data = ch_data
                mn, mx = np.nanmin(data), np.nanmax(data)
                if not np.isfinite(mn) or not np.isfinite(mx):
                    mn, mx = -1, 1
                if mn == mx:
                    ax.set_ylim(mn - 1, mx + 1)
                else:
                    margin = (mx - mn) * 0.2
                    ax.set_ylim(mn - margin, mx + margin)
        self.draw()

class BarCanvas(FigureCanvas):
    def __init__(self, figsize=(4,2)):
        self.fig = Figure(figsize=figsize, tight_layout=True)
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)

    def plot_confidences(self, labels, probs):
        self.ax.clear()
        self.ax.bar(labels, probs, color=["#1f77b4", "#ff7f0e", "#aec7e8", "#ffbb78"])
        self.ax.set_ylim(0,1)
        self.ax.set_ylabel("Probability")
        self.ax.set_title("Valence/Arousal Confidence")
        self.fig.autofmt_xdate(rotation=0)
        self.draw()

class WheelCanvas(FigureCanvas):
    def __init__(self, figsize=(3,3)):
        # Labels by ID (not wedge order)
        self.labels_by_id = [QUAD_ID_TO_NAME[i].split(" (")[0] for i in range(4)]
        # Map ID -> wedge index for rendering order [TR, BR, BL, TL]
        self.id_to_wedge = {0: 0, 1: 3, 2: 2, 3: 1}
        self.fig = Figure(figsize=figsize, tight_layout=True)
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111, aspect='equal')

    def plot_wheel(self, highlight_id=None, highlight_label=None):
        self.ax.clear()

        # Place wedges in TR, BR, BL, TL order -> [id0, id3, id2, id1]
        labels_wedge_order = [
            self.labels_by_id[0],
            self.labels_by_id[3],
            self.labels_by_id[2],
            self.labels_by_id[1],
        ]
        probs = np.ones(4) / 4.0
        wedges, _ = self.ax.pie(
            probs,
            labels=labels_wedge_order,
            startangle=90,          # top
            counterclock=False,     # clockwise: TR, BR, BL, TL
            textprops={'fontsize': 9}
        )

        # Crosshair axes
        self.ax.plot([-1.05, 1.05], [0, 0], color='gray', lw=1)
        self.ax.plot([0, 0], [-1.05, 1.05], color='gray', lw=1)
        # Axis labels
        self.ax.text(0, 1.12, "High Arousal", ha='center', va='bottom', fontsize=9)
        self.ax.text(0, -1.12, "Low Arousal", ha='center', va='top', fontsize=9)
        self.ax.text(1.12, 0, "High Valence", ha='left', va='center', fontsize=9)
        self.ax.text(-1.12, 0, "Low Valence", ha='right', va='center', fontsize=9)

        # Highlight by id or label
        wedge_idx = None
        if highlight_id is not None:
            wedge_idx = self.id_to_wedge.get(int(highlight_id), None)
        elif highlight_label is not None:
            try:
                lbl = highlight_label.strip().lower()
                id_by_lbl = {n.lower(): i for i, n in enumerate(self.labels_by_id)}
                q_id = id_by_lbl.get(lbl, None)
                wedge_idx = self.id_to_wedge.get(q_id, None)
            except Exception:
                wedge_idx = None
        if wedge_idx is not None and 0 <= wedge_idx < 4:
            wedges[wedge_idx].set_edgecolor('black')
            wedges[wedge_idx].set_linewidth(2.5)

        self.ax.set_title("Emotion Wheel (SAM/Russell)")
        self.ax.set_xlim(-1.2, 1.2); self.ax.set_ylim(-1.2, 1.2)
        self.draw()


# -----------------------------------------------
# SHAP Popup Window
# -----------------------------------------------
class ShapWindow(QWidget):
    def __init__(self, shap_fig):
        super().__init__()
        self.setWindowTitle("SHAP Explainability")
        self.resize(900, 700)

        layout = QVBoxLayout()
        self.setLayout(layout)

        # Convert SHAP fig to canvas
        self.canvas = FigureCanvas(shap_fig)
        layout.addWidget(self.canvas)


# -----------------------------
# Main GUI
# -----------------------------
class EEGDemoGUI(QWidget):
    def __init__(self, model_path, data_dir):
        super().__init__()
        self.setWindowTitle("EEG Emotion Demo - MVGT-BiLSTM")
        self.setGeometry(100,50,1400,900)

        # Data/config
        self.data_dir = data_dir
        self.subjects = ["s31", "s32"]
        self.nchannels = 32
        self.nfeatures = 17     # model expects (batch, 32, 17)
        self.sampling_rate = 128
        self.buffer_len_seconds = 4
        self.buffer_len = int(self.sampling_rate * self.buffer_len_seconds)
        self.play_index = 0
        self.playing = False
        self.eeg_data = None   # (n_epochs, 32, 17)
        self.labels = None
        self.last_prediction = None

        # SHAP placeholders
        self.shap_explainer = None
        self.shap_background = None
        self.shap_target_head = 0  # 0=valence, 1=arousal — toggled by button
        self.shap_popup = None     # keep reference to popup

        # Model (loaded without compile; we handle predict robustly)
        self.model = load_model(
            model_path,
            custom_objects={
                'SpatialEncoding': SpatialEncoding,
                'GraphMultiHeadAttention': GraphMultiHeadAttention
            }
        )

        # GUI
        self.build_ui()

        # Timer for epoch playback
        self.timer = QTimer()
        self.timer.setInterval(500)
        self.timer.timeout.connect(self.on_timer)

        # Load default subject immediately
        if self.subjects:
            self.subj_combo.setCurrentIndex(0)
            self.load_subject(self.subj_combo.currentText())

    def build_ui(self):
        main_layout = QHBoxLayout()
        left_col = QVBoxLayout()
        right_col = QVBoxLayout()
        self.right_col = right_col

        # ========== Left: Data and plots ==========
        subj_group = QGroupBox("Data Selection")
        subj_layout = QVBoxLayout()
        self.subj_combo = QComboBox()
        self.subj_combo.addItems(self.subjects)
        self.subj_combo.currentTextChanged.connect(self.load_subject)
        subj_layout.addWidget(QLabel("Select Subject:"))
        subj_layout.addWidget(self.subj_combo)

        controls_row = QHBoxLayout()
        self.prev_btn = QPushButton("⟵ Prev")
        self.next_btn = QPushButton("Next ⟶")
        self.prev_btn.clicked.connect(self.on_prev)
        self.next_btn.clicked.connect(self.on_next)
        controls_row.addWidget(self.prev_btn)
        controls_row.addWidget(self.next_btn)

        win_row = QHBoxLayout()
        win_row.addWidget(QLabel("Prediction window (epochs):"))
        self.window_spin = QSpinBox()
        self.window_spin.setRange(1, 50)
        self.window_spin.setValue(5)
        win_row.addWidget(self.window_spin)

        subj_layout.addLayout(controls_row)
        subj_layout.addLayout(win_row)
        subj_group.setLayout(subj_layout)
        left_col.addWidget(subj_group)

        # Plot area with channel labels
        self.canvas = MultiChannelCanvas(
            nchannels=self.nchannels, rows=8, cols=4, channel_names=CHANNEL_NAMES_32
        )
        left_col.addWidget(self.canvas, stretch=6)

        # Player controls
        player_layout = QHBoxLayout()
        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self.toggle_play)
        self.pause_btn = QPushButton("Pause")
        self.pause_btn.clicked.connect(self.pause_play)
        self.predict_btn = QPushButton("Predict")
        self.predict_btn.clicked.connect(self.on_predict)
        # SHAP explainability button (cycles between valence/arousal)
        self.shap_btn = QPushButton("Explain: Valence")
        self.shap_btn.clicked.connect(self.on_shap_btn)
        player_layout.addWidget(self.play_btn)
        player_layout.addWidget(self.pause_btn)
        player_layout.addWidget(self.predict_btn)
        player_layout.addWidget(self.shap_btn)
        left_col.addLayout(player_layout)

        # ========== Right: Info and predictions ==========
        info_group = QGroupBox("EEG Info")
        info_layout = QVBoxLayout()
        self.file_label = QLabel("No EEG loaded")
        self.epoch_label = QLabel("Epoch: - / -")
        self.gt_label = QLabel("Ground truth (Y): n/a")
        info_layout.addWidget(self.file_label)
        info_layout.addWidget(self.epoch_label)
        info_layout.addWidget(self.gt_label)
        info_group.setLayout(info_layout)
        right_col.addWidget(info_group)

        wheel_group = QGroupBox("Emotion Wheel (Quadrants)")
        wheel_layout = QVBoxLayout()
        self.wheel = WheelCanvas()
        self.wheel.plot_wheel()  # default
        self.mapping_label = QLabel(
            "Orientation: TR=0:Excited (HV+HA), TL=1:Tense (LV+HA), BL=2:Sad (LV+LA), BR=3:Calm (HV+LA)"
        )
        wheel_layout.addWidget(self.wheel)
        wheel_layout.addWidget(self.mapping_label)
        wheel_group.setLayout(wheel_layout)
        right_col.addWidget(wheel_group, stretch=2)

        # --- SHAP Heatmap Canvas inserted between wheel and prediction group ---
        self.shap_fig = Figure(figsize=(4, 3), tight_layout=True)
        self.shap_canvas = FigureCanvas(self.shap_fig)
        ax0 = self.shap_fig.add_subplot(111)
        ax0.text(0.5, 0.5, "SHAP heatmap\n(Press Explain (SHAP))", ha='center', va='center', fontsize=9)
        ax0.axis('off')
        self.shap_canvas.draw()
        right_col.addWidget(self.shap_canvas, stretch=1)

        pred_group = QGroupBox("Prediction")
        pred_layout = QVBoxLayout()
        self.bar = BarCanvas()
        pred_layout.addWidget(self.bar, stretch=1)
        self.model_status = QLabel("Model: loaded")
        pred_layout.addWidget(self.model_status)
        self.window_used_label = QLabel("Prediction used epochs: -")
        pred_layout.addWidget(self.window_used_label)
        pred_group.setLayout(pred_layout)
        right_col.addWidget(pred_group, stretch=1)

        main_layout.addLayout(left_col, stretch=7)
        main_layout.addLayout(right_col, stretch=3)
        self.setLayout(main_layout)

    # -------------------------
    # Load preprocessed EEG for selected subject
    # -------------------------
    def load_subject(self, subj_id):
        try:
            X_path = os.path.join(self.data_dir, f"test_{subj_id}_X.npy")
            Y_path = os.path.join(self.data_dir, f"test_{subj_id}_Y.npy")
            X = np.load(X_path)
            Y = np.load(Y_path, allow_pickle=True)

            X = self._coerce_X_layout(X)  # ensure (n_epochs, 32, 17)
            self.eeg_data = X
            self.labels = Y

            n_epochs = X.shape[0]
            per_epoch_shape = tuple(X.shape[1:])
            self.file_label.setText(f"EEG: subject {subj_id} | epochs: {n_epochs} | epoch shape: {per_epoch_shape}")
            self.play_index = 0

            # Initialize SHAP explainer (non-destructive; ignore errors)
            try:
                # pick up to 20 background epochs
                n_bg = min(20, self.eeg_data.shape[0])
                idx = np.random.choice(self.eeg_data.shape[0], size=n_bg, replace=False)
                self.shap_background = self.eeg_data[idx]  # (n_bg, 32, 17)

                # Prepare background for model (ensures batch dim)
                bg = self._prepare_for_model(self.shap_background)
                if bg.ndim == 2:
                    bg = np.expand_dims(bg, axis=0)

                # Build single-output wrapper for the currently selected head
                self._build_shap_explainer(bg, head=self.shap_target_head)
                print("SHAP (GradientExplainer) ready.")
            except Exception as e:
                self.shap_explainer = None
                print("SHAP init error:", e)

            # Initial draw
            self.update_plot()
            self.update_info_labels()

            # Optional: console sanity
            try:
                print("X coerced shape:", self.eeg_data.shape)
                arr = np.asarray(self.labels)
                if arr.ndim == 1:
                    vals, cnt = np.unique(arr, return_counts=True)
                    print("GT id distribution:", dict(zip(vals.tolist(), cnt.tolist())))
                elif arr.ndim >= 2:
                    if arr.shape[-1] == 4:
                        ids = np.argmax(arr, axis=-1)
                        vals, cnt = np.unique(ids, return_counts=True)
                        print("GT one-hot->id distribution:", dict(zip(vals.tolist(), cnt.tolist())))
                    elif arr.shape[-1] >= 2:
                        combos, cnt = np.unique(arr[..., :2], axis=0, return_counts=True)
                        print("GT V/A combos:", combos, cnt)
            except Exception:
                pass

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not load demo subject {subj_id}:\n{e}")

    # -------------------------
    # Shape coercion helpers
    # -------------------------
    def _coerce_X_layout(self, X):
        """
        Ensure X is 3D and shaped (n_epochs, 32, 17).
        Accepts (n_epochs, 17, 32) and swaps axes.
        """
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 3:
            raise ValueError(f"Expected 3D X (n_epochs, C, T). Got {X.shape}")
        n, d1, d2 = X.shape
        # quick auto-fix: (n, features, channels) -> swap
        if d1 == self.nfeatures and d2 == self.nchannels:
            return np.swapaxes(X, 1, 2)
        if d1 == self.nchannels and d2 == self.nfeatures:
            return X
        if d2 == self.nchannels and d1 == self.nfeatures:
            return np.swapaxes(X, 1, 2)  # (n, 17, 32) -> (n, 32, 17)
        # If one dim is 32 but the other is not 17, stop with clear error
        if d1 == self.nchannels and d2 != self.nfeatures:
            raise ValueError(f"Expected features dim=17, got {d2}.")
        if d2 == self.nchannels and d1 != self.nfeatures:
            raise ValueError(f"Expected features dim=17 (before swap), got {d1}.")
        raise ValueError(f"Expected shapes (n,32,17) or (n,17,32), got {X.shape}")

    # -------------------------
    # Helpers for plotting and info
    # -------------------------
    def _epoch_to_canvas(self, epoch2d):
        """
        For plotting we want (channels, features).
        """
        if epoch2d.ndim != 2:
            return None
        if epoch2d.shape[0] == self.nchannels:
            return epoch2d  # (32, 17)
        if epoch2d.shape[1] == self.nchannels:
            return epoch2d.T  # (17, 32) -> (32, 17)
        return epoch2d

    def update_plot(self):
        if self.eeg_data is None or self.eeg_data.size == 0:
            return
        if self.play_index >= self.eeg_data.shape[0]:
            self.play_index = 0
        epoch = self.eeg_data[self.play_index]  # (32, 17)
        canvas_epoch = self._epoch_to_canvas(epoch)
        self.canvas.update_data(canvas_epoch)
        subj = self.subj_combo.currentText()
        n_epochs = self.eeg_data.shape[0]
        self.canvas.set_header(f"Subject {subj} • Epoch {self.play_index+1}/{n_epochs}")
        self.update_info_labels()

    def update_info_labels(self):
        if self.eeg_data is None:
            self.epoch_label.setText("Epoch: - / -")
            self.gt_label.setText("Ground truth (Y): n/a")
            return
        n_epochs = self.eeg_data.shape[0]
        self.epoch_label.setText(f"Epoch: {self.play_index+1} / {n_epochs}")
        try:
            y = self.labels[self.play_index]
            self.gt_label.setText(decode_quad_y(y))
        except Exception:
            self.gt_label.setText("Ground truth (Y): n/a")

    # -------------------------
    # Play / Timer / Navigation
    # -------------------------
    def toggle_play(self):
        if self.eeg_data is None:
            QMessageBox.warning(self, "Play", "No EEG loaded.")
            return
        if not self.playing:
            self.playing = True
            self.timer.start()
            self.play_btn.setText("Playing...")
        else:
            self.playing = False
            self.timer.stop()
            self.play_btn.setText("Play")

    def pause_play(self):
        self.playing = False
        self.timer.stop()
        self.play_btn.setText("Play")

    def on_prev(self):
        if self.eeg_data is None:
            return
        self.play_index = (self.play_index - 1) % self.eeg_data.shape[0]
        self.update_plot()

    def on_next(self):
        if self.eeg_data is None:
            return
        self.play_index = (self.play_index + 1) % self.eeg_data.shape[0]
        self.update_plot()

    def on_timer(self):
        if self.eeg_data is None:
            return
        self.play_index = (self.play_index + 1) % self.eeg_data.shape[0]
        self.update_plot()

    # -------------------------
    # Prediction utilities
    # -------------------------
    def _prepare_for_model(self, X):
        """
        Ensure input is (batch, 32, 17). Accepts (batch, 17, 32) and swaps axes.
        """
        X = np.asarray(X, dtype=np.float32)
        orig_shape = X.shape
        if X.ndim == 2:
            X = X[None, ...]  # single epoch -> batch 1
        if X.ndim != 3:
            raise ValueError(f"Model expects 3D input (batch,32,17). Got {orig_shape}")
        b, d1, d2 = X.shape
        if d1 == self.nchannels and d2 == self.nfeatures:
            return X
        if d2 == self.nchannels and d1 == self.nfeatures:
            return np.swapaxes(X, 1, 2)
        raise ValueError(f"Cannot adapt input to (batch,32,17). Got {orig_shape} -> {X.shape}")

    def _softmax_np(self, x, axis=-1):
        x = np.asarray(x, dtype=np.float32)
        x_max = np.max(x, axis=axis, keepdims=True)
        e = np.exp(x - x_max)
        return e / np.sum(e, axis=axis, keepdims=True)

    def _va_from_quadrant_probs(self, q):
        """
        q: (..., 4) in your order:
           0:Excited(HV+HA), 1:Tense(LV+HA), 2:Sad(LV+LA), 3:Calm(HV+LA)
        Returns continuous:
          v = P(HV) = q0 + q3
          a = P(HA) = q0 + q1
          q_probs = normalized probs (..., 4)
        """
        q = np.asarray(q)
        if q.ndim == 1:
            q = q[None, :]
        # Normalize if logits or not summing to 1
        if (np.any(q < 0) or np.any(q > 1)) or (not np.allclose(q.sum(-1), 1, atol=1e-3)):
            q = self._softmax_np(q, axis=-1)
        v = q[..., 0] + q[..., 3]
        a = q[..., 0] + q[..., 1]
        return v, a, q

    def _predict_safe(self, X):
        """
        Robust prediction that handles multiple output formats:
          - Two binary heads -> returns [valence_probs, arousal_probs]
          - Single 4-class softmax -> returns [q_logits] (single array)
          - dict outputs -> tries to map keys
        Returns the raw model outputs (converted to list where appropriate).
        """
        X_prepared = self._prepare_for_model(X)  # ensures batch dimension
        outputs = self.model.predict(X_prepared, verbose=0)

        # Normalize outputs to list-like structure for downstream parsing
        if isinstance(outputs, dict):
            # try keys for valence/arousal first, else return values list
            keys_map = {k.lower(): k for k in outputs.keys()}
            v_key = next((orig for lk, orig in keys_map.items() if "val" in lk), None)
            a_key = next((orig for lk, orig in keys_map.items() if "aro" in lk or "arous" in lk or lk == "a"), None)
            if v_key is not None and a_key is not None:
                return [np.asarray(outputs[v_key]), np.asarray(outputs[a_key])]
            # fallback: list of values
            return [np.asarray(v) for v in outputs.values()]

        if isinstance(outputs, (list, tuple)):
            return [np.asarray(o) for o in outputs]

        # single numpy array
        arr = np.asarray(outputs)
        return [arr]

    def _parse_valence_arousal(self, outputs_list):
        """
        Parse outputs_list into (valence_probs_array, arousal_probs_array)
        outputs_list is a list of arrays (as returned by _predict_safe).
        Supports:
         - two-head model: [val, ar]
         - single 4-class output: [q_logits] where we map to v/a
        """
        # Two-head case
        if len(outputs_list) >= 2:
            v_raw = outputs_list[0].squeeze()
            a_raw = outputs_list[1].squeeze()
            # ensure arrays are 1D (batch)
            if v_raw.ndim > 0:
                v_raw = v_raw.ravel()
            if a_raw.ndim > 0:
                a_raw = a_raw.ravel()
            return v_raw.astype(float), a_raw.astype(float)

        # Single array case (maybe 4-class)
        only = outputs_list[0]
        only = np.asarray(only)
        if only.ndim == 2 and only.shape[-1] == 4:
            # batch,4 -> compute continuous v,a
            v_cont, a_cont, q_probs = self._va_from_quadrant_probs(only)
            return v_cont.squeeze().astype(float), a_cont.squeeze().astype(float)

        # Single 2-dim (batch,2) -> interpret as [V, A]
        if only.ndim == 2 and only.shape[-1] == 2:
            return only[..., 0].astype(float).squeeze(), only[..., 1].astype(float).squeeze()

        # Fallback: treat single scalar as valence only
        raise ValueError(f"Cannot parse model outputs to valence/arousal. shape={getattr(only, 'shape', None)}")

    # -------------------------
    # SHAP helpers
    # -------------------------
    def _build_shap_explainer(self, bg_array, head=0):
        """
        Build GradientExplainer for a single output head:
        - bg_array: background array already prepared (batch,32,17)
        - head: 0 (valence) or 1 (arousal) or None for 4-class single head
        """
        try:
            # If the model has multiple outputs, wrap to single head
            # If the model has a single output (4-class), head should be None or 0
            model_outputs = getattr(self.model, "outputs", None)
            if model_outputs is not None and isinstance(model_outputs, (list, tuple)) and len(model_outputs) >= 2:
                # create a single-output wrapper for desired head
                single_output_model = tf.keras.Model(inputs=self.model.input, outputs=self.model.outputs[head])
            else:
                # single-output model (keep as-is)
                single_output_model = self.model

            # Ensure background has batch dim
            bg = np.asarray(bg_array)
            if bg.ndim == 2:
                bg = np.expand_dims(bg, axis=0)

            # Use GradientExplainer (works with TF models)
            self.shap_explainer = shap.GradientExplainer(single_output_model, bg)
            self.shap_target_head = head
        except Exception as e:
            self.shap_explainer = None
            raise

    def on_shap_btn(self):
        """
        Cycle SHAP target head between valence (0) and arousal (1),
        rebuild explainer for new head (if possible), and run explanation.
        """
        # if the loaded model is single-output (4-class), we keep single explainer
        # else toggle between 0 and 1
        try:
            model_outputs = getattr(self.model, "outputs", None)
            if model_outputs is not None and isinstance(model_outputs, (list, tuple)) and len(model_outputs) >= 2:
                # toggle head 0/1
                self.shap_target_head = 1 - self.shap_target_head
                label = "Valence" if self.shap_target_head == 0 else "Arousal"
                self.shap_btn.setText(f"Explain: {label}")
                # rebuild explainer with same background if available
                if self.shap_background is not None:
                    bg = self._prepare_for_model(self.shap_background)
                    if bg.ndim == 2:
                        bg = np.expand_dims(bg, axis=0)
                    try:
                        self._build_shap_explainer(bg, head=self.shap_target_head)
                    except Exception as e:
                        print("SHAP rebuild error:", e)
            else:
                # single-output model, do nothing
                self.shap_btn.setText("Explain: Model (single-output)")
            # call explanation immediately
            self.on_shap()
        except Exception as e:
            QMessageBox.warning(self, "SHAP", f"Could not toggle SHAP head:\n{e}")

    def on_shap(self):
        """
        Compute SHAP for the current epoch and draw heatmap.
        Robust handling of shapes and SHAP output types.
        """
        if self.eeg_data is None:
            QMessageBox.warning(self, "SHAP", "No EEG loaded.")
            return
        if self.shap_explainer is None:
            QMessageBox.warning(self, "SHAP", "SHAP explainer not initialized (did subject load fail?).")
            return

        try:
            epoch = self.eeg_data[self.play_index]  # (32,17)
            epoch_ = self._prepare_for_model(epoch)  # (1,32,17)
            if epoch_.ndim == 2:
                epoch_ = np.expand_dims(epoch_, axis=0)

            # shap_values may return list or array
            shap_vals = self.shap_explainer.shap_values(epoch_)

            # If GradientExplainer returns list (per-class), select appropriate element:
            # For single-output wrapper shap_values often returns a list with a single array (so take [0])
            if isinstance(shap_vals, list):
                # The explainer was built on a single-output wrapper, so shap_vals[0] is expected
                shap_arr = np.asarray(shap_vals[0])
            else:
                shap_arr = np.asarray(shap_vals)

            # shap_arr could be (1,32,17) or (32,17) or (batch, channels, features)
            sv = None
            # Try multiple possible shapes
            if shap_arr.ndim == 3:
                # common: (batch, channels, features)
                if shap_arr.shape[0] >= 1 and shap_arr.shape[1] == self.nchannels and shap_arr.shape[2] == self.nfeatures:
                    sv = shap_arr[0]
                else:
                    # try squeeze then match
                    s = np.squeeze(shap_arr)
                    if s.ndim == 2 and s.shape == (self.nchannels, self.nfeatures):
                        sv = s
                    elif s.ndim == 2 and s.shape == (self.nfeatures, self.nchannels):
                        sv = s.T
            elif shap_arr.ndim == 2:
                if shap_arr.shape == (self.nchannels, self.nfeatures):
                    sv = shap_arr
                elif shap_arr.shape == (self.nfeatures, self.nchannels):
                    sv = shap_arr.T
                else:
                    # maybe (batch, flattened) or other -> try to reshape if size matches
                    if shap_arr.size == self.nchannels * self.nfeatures:
                        sv = shap_arr.reshape(self.nchannels, self.nfeatures)
            else:
                # fallback: squeeze then check
                s = np.squeeze(shap_arr)
                if s.ndim == 2 and s.shape == (self.nchannels, self.nfeatures):
                    sv = s
                elif s.ndim == 2 and s.shape == (self.nfeatures, self.nchannels):
                    sv = s.T

            if sv is None:
                QMessageBox.warning(self, "SHAP", f"Unexpected SHAP shape: {shap_arr.shape}")
                return

            # final safety reshape if flattened
            if sv.shape != (self.nchannels, self.nfeatures):
                try:
                    if sv.size == self.nchannels * self.nfeatures:
                        sv = sv.reshape(self.nchannels, self.nfeatures)
                except Exception:
                    pass

            # # Update the small embedded SHAP panel (keeps main UI showing a tiny preview)
            # try:
            #     self.shap_fig.clear()
            #     ax_preview = self.shap_fig.add_subplot(111)
            #     im = ax_preview.imshow(sv, aspect='auto')
            #     self.shap_fig.colorbar(im, ax=ax_preview)
            #     ax_preview.set_xticks([])
            #     ax_preview.set_yticks([])
            #     self.shap_canvas.draw()
            # except Exception as e:
            #     # non-fatal: continue to popup even if embedded preview fails
            #     print("Warning: embedded SHAP preview draw error:", e)

            # Now create and show popup with its own Figure (avoids sharing canvas objects)
            self._show_shap_popup(sv)

        except Exception as e:
            QMessageBox.critical(self, "SHAP Error", f"Failed to compute SHAP:\n{e}")

    def _show_shap_popup(self, shap_matrix):
        """
        Draw shap_matrix into a new Figure and display in a popup window.
        This avoids passing AxesImage or reusing canvas objects.
        """
        try:
            # Create a dedicated figure for popup
            fig_popup = Figure(figsize=(8, 6), tight_layout=True)
            ax = fig_popup.add_subplot(111)
            im = ax.imshow(shap_matrix, aspect='auto')
            cbar = fig_popup.colorbar(im, ax=ax)
            cbar.set_label("SHAP value")

            # x ticks -> feature names
            x_ticks = np.arange(shap_matrix.shape[1])
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(FEATURE_NAMES_17, fontsize=7, rotation=45)

            # y ticks -> channel names
            y_ticks = np.arange(min(len(CHANNEL_NAMES_32), shap_matrix.shape[0]))
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(CHANNEL_NAMES_32[:shap_matrix.shape[0]], fontsize=7)

            # Title: dynamic (Valence or Arousal)
            head_name = "Valence" if self.shap_target_head == 0 else "Arousal"
            ax.set_title(f"SHAP Heatmap for {head_name} — Epoch {self.play_index+1}")
            ax.set_xlabel("Features")
            ax.set_ylabel("Channels")

            self.shap_popup = ShapWindow(fig_popup)
            self.shap_popup.show()
        except Exception as e:
            print("Error creating SHAP popup:", e)

    # -------------------------
    # Aggregated Prediction over multiple epochs
    # -------------------------
    def on_predict(self):
        if self.eeg_data is None:
            QMessageBox.warning(self, "Predict", "No EEG loaded")
            return

        window_size = int(self.window_spin.value())
        n_epochs = self.eeg_data.shape[0]
        start_idx = self.play_index
        end_idx = min(self.play_index + window_size, n_epochs)
        if end_idx <= start_idx:
            QMessageBox.warning(self, "Predict", "Invalid prediction window (no epochs).")
            return

        epochs_window = self.eeg_data[start_idx:end_idx]
        self.window_used_label.setText(f"Prediction used epochs: {start_idx} .. {end_idx-1} (count={end_idx - start_idx})")

        try:
            outputs_list = self._predict_safe(epochs_window)  # list of arrays
            preds_v, preds_a = self._parse_valence_arousal(outputs_list)
        except Exception as e:
            QMessageBox.critical(self, "Predict Error", f"Prediction failed:\n{e}")
            return

        # Aggregate continuous P(High V), P(High A): these are arrays of shape (batch,)
        avg_v = float(np.mean(preds_v))
        avg_a = float(np.mean(preds_a))
        # Direct binary decision
        v_bit = 1 if avg_v > 0.5 else 0
        a_bit = 1 if avg_a > 0.5 else 0
        q_pred = VA_TO_QUAD_ID[(v_bit, a_bit)]
        label = QUAD_ID_TO_NAME[q_pred].split(" (")[0]

        # Update visuals
        probs = [avg_v, avg_a, 1-avg_v, 1-avg_a]
        self.wheel.plot_wheel(highlight_id=q_pred)
        self.bar.plot_confidences(["V_H","A_H","V_L","A_L"], probs)
        self.model_status.setText(
            f"Predicted: {label} (Quadrant {q_pred}) | "
            f"Valence: {avg_v:.3f} ({'High' if v_bit else 'Low'}) | "
            f"Arousal: {avg_a:.3f} ({'High' if a_bit else 'Low'})"
        )

        # Store last prediction for saving/logging
        self.last_prediction = {
            'label': label,
            'pred_quadrant_id': int(q_pred),
            'probs': {'V_H': float(avg_v), 'A_H': float(avg_a), 'V_L': float(1-avg_v), 'A_L': float(1-avg_a)},
            'window': {'start_epoch': int(start_idx), 'end_epoch_inclusive': int(end_idx-1)},
            'timestamp': datetime.now().isoformat()
        }

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    model_path = "1results_freeze_unfreeze_mvgt_bilstm/final_overall_model.h5"
    data_dir = "data/processed"
    app = QApplication(sys.argv)
    window = EEGDemoGUI(model_path, data_dir)
    window.show()
    sys.exit(app.exec())

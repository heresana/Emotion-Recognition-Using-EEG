# eeg_demo_gui_mvgt.py
import sys, os
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

import tensorflow as tf
from tensorflow.keras.models import load_model

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.custom_layers import SpatialEncoding, GraphMultiHeadAttention

# -----------------------------
# Channel names (32-ch DEAP-like)
# -----------------------------
CHANNEL_NAMES_32 = [
    "Fp1","AF3","F3","F7","FC5","FC1","C3","T7",
    "CP5","CP1","P3","P7","PO3","O1","Oz","Pz",
    "Fp2","AF4","Fz","F4","F8","FC6","FC2","Cz",
    "C4","T8","CP6","CP2","P4","P8","PO4","O2"
]

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
        # chunk_buffer: (nchannels, time)
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
    def __init__(self, labels=None, figsize=(3,3)):
        self.labels = labels or ["Happy","Sad","Angry","Fear","Calm","Disgust","Surprise","Neutral"]
        self.fig = Figure(figsize=figsize, tight_layout=True)
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111, aspect='equal')

    def plot_wheel(self, highlight_label=None):
        self.ax.clear()
        n = len(self.labels)
        probs = np.ones(n)/n
        wedges, _ = self.ax.pie(probs, labels=self.labels, startangle=90, textprops={'fontsize': 8})
        if highlight_label in self.labels:
            idx = self.labels.index(highlight_label)
            wedges[idx].set_edgecolor('black'); wedges[idx].set_linewidth(2.5)
        self.ax.set_title("Emotion Wheel (Quadrants)")
        self.draw()

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
        self.sampling_rate = 128
        self.buffer_len_seconds = 4
        self.buffer_len = int(self.sampling_rate * self.buffer_len_seconds)
        self.play_index = 0
        self.playing = False
        self.eeg_data = None   # expected: (n_epochs, time, channels) or (n_epochs, channels, time)
        self.labels = None     # whatever is saved in *_Y.npy (optional)
        self.last_prediction = None

        # Model (loaded without compile; we handle predict robustly)
        self.model = load_model(
            model_path,
            compile=False,
            custom_objects={
                'SpatialEncoding': SpatialEncoding,
                'GraphMultiHeadAttention': GraphMultiHeadAttention
            }
        )

        # GUI
        self.build_ui()

        # Timer for epoch playback
        self.timer = QTimer()
        self.timer.setInterval(500)  # slower to make transitions visible
        self.timer.timeout.connect(self.on_timer)

        # Load default subject immediately so "No EEG loaded" never shows
        if self.subjects:
            self.subj_combo.setCurrentIndex(0)
            self.load_subject(self.subj_combo.currentText())

    def build_ui(self):
        main_layout = QHBoxLayout()
        left_col = QVBoxLayout()
        right_col = QVBoxLayout()

        # ========== Left: Data and plots ==========
        # Subject selection group
        subj_group = QGroupBox("Data Selection")
        subj_layout = QVBoxLayout()
        self.subj_combo = QComboBox()
        self.subj_combo.addItems(self.subjects)
        self.subj_combo.currentTextChanged.connect(self.load_subject)
        subj_layout.addWidget(QLabel("Select Subject:"))
        subj_layout.addWidget(self.subj_combo)

        # Epoch controls
        controls_row = QHBoxLayout()
        self.prev_btn = QPushButton("⟵ Prev")
        self.next_btn = QPushButton("Next ⟶")
        self.prev_btn.clicked.connect(self.on_prev)
        self.next_btn.clicked.connect(self.on_next)
        controls_row.addWidget(self.prev_btn)
        controls_row.addWidget(self.next_btn)

        # Window size for prediction
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
        player_layout.addWidget(self.play_btn)
        player_layout.addWidget(self.pause_btn)
        player_layout.addWidget(self.predict_btn)
        left_col.addLayout(player_layout)

        # ========== Right: Info and predictions ==========
        # EEG info group
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

        # Emotion wheel
        wheel_group = QGroupBox("Emotion Wheel (Quadrants)")
        wheel_layout = QVBoxLayout()
        self.wheel = WheelCanvas(labels=["Happy","Sad","Angry","Fear","Calm","Disgust","Surprise","Neutral"])
        self.wheel.plot_wheel()  # default
        wheel_layout.addWidget(self.wheel)
        self.mapping_label = QLabel("Mapping: HV+HA=Happy, HV+LA=Calm, LV+HA=Fear, LV+LA=Sad")
        wheel_layout.addWidget(self.mapping_label)
        wheel_group.setLayout(wheel_layout)
        right_col.addWidget(wheel_group, stretch=2)

        # Bar chart and model status
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

            # Ensure float32, int or whatever as appropriate
            X = np.asarray(X).astype(np.float32)
            self.eeg_data = X
            self.labels = Y

            n_epochs = X.shape[0]
            per_epoch_shape = tuple(X.shape[1:])
            self.file_label.setText(f"EEG: subject {subj_id} | epochs: {n_epochs} | epoch shape: {per_epoch_shape}")
            self.play_index = 0

            # Initial draw
            self.update_plot()
            self.update_info_labels()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not load demo subject {subj_id}:\n{e}")

    # -------------------------
    # Helpers for plotting and info
    # -------------------------
    def _epoch_to_canvas(self, epoch2d):
        """
        Normalize epoch to (channels, time) for plotting.
        Accepts (time, channels) or (channels, time).
        """
        if epoch2d.ndim != 2:
            return None
        t, c = epoch2d.shape[0], epoch2d.shape[1]

        # Prefer last dimension to be channels for training data
        # For canvas, we want (channels, time)
        if epoch2d.shape[0] == self.nchannels and epoch2d.shape[1] != self.nchannels:
            return epoch2d  # already (channels, time)
        if epoch2d.shape[1] == self.nchannels:
            return epoch2d.T  # (time, channels) -> (channels, time)

        # Fallback: choose the dimension closer to nchannels as channel axis
        if abs(epoch2d.shape[0] - self.nchannels) < abs(epoch2d.shape[1] - self.nchannels):
            # assume 0 is channels
            return epoch2d[:self.nchannels, :]
        else:
            # assume 1 is channels
            return epoch2d[:, :self.nchannels].T

    def update_plot(self):
        if self.eeg_data is None or self.eeg_data.size == 0:
            return
        if self.play_index >= self.eeg_data.shape[0]:
            self.play_index = 0
        epoch = self.eeg_data[self.play_index]  # 2D
        # Prepare data for canvas
        canvas_epoch = self._epoch_to_canvas(epoch)
        self.canvas.update_data(canvas_epoch)
        subj = self.subj_combo.currentText()
        n_epochs = self.eeg_data.shape[0]
        self.canvas.set_header(f"Subject {subj} • Epoch {self.play_index+1}/{n_epochs}")

        # Update info labels
        self.update_info_labels()

    def update_info_labels(self):
        if self.eeg_data is None:
            self.epoch_label.setText("Epoch: - / -")
            self.gt_label.setText("Ground truth (Y): n/a")
            return
        n_epochs = self.eeg_data.shape[0]
        self.epoch_label.setText(f"Epoch: {self.play_index+1} / {n_epochs}")
        # Show whatever is present in Y
        try:
            y = self.labels[self.play_index]
            if isinstance(y, np.ndarray) and y.ndim > 0:
                y_text = np.array2string(y, precision=2, separator=",")
            else:
                y_text = str(y)
            self.gt_label.setText(f"Ground truth (Y): {y_text}")
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
    def _predict_safe(self, X):
        """
        Robust prediction that handles:
        - tf.function returning empty outputs (falls back to eager)
        - multiple output formats (2 heads; dict; single output with 2 dims)
        Returns numpy arrays (preds_v, preds_a) of shape (n_window,)
        """
        # First try standard Keras predict
        try:
            outputs = self.model.predict(X, verbose=0)
        except Exception as e:
            # Fallback to eager
            try:
                tf.config.run_functions_eagerly(True)
            except Exception:
                pass
            y = self.model(X, training=False)
            outputs = y

            # Convert to numpy
            def to_np(o):
                if isinstance(o, tf.Tensor):
                    return o.numpy()
                return o
            if isinstance(outputs, (list, tuple)):
                outputs = [to_np(o) for o in outputs]
            elif isinstance(outputs, dict):
                outputs = {k: to_np(v) for k, v in outputs.items()}
            else:
                outputs = to_np(outputs)

        # Parse outputs into (valence, arousal)
        preds_v, preds_a = self._parse_valence_arousal(outputs)
        return np.asarray(preds_v).astype(float), np.asarray(preds_a).astype(float)

    def _parse_valence_arousal(self, outputs):
        """
        Accepts:
        - list/tuple of two arrays [valence, arousal]
        - dict with keys ('valence','arousal') or ('v','a'), etc.
        - single array with last dim == 2 => [:,0] valence, [:,1] arousal
        """
        # list/tuple
        if isinstance(outputs, (list, tuple)):
            if len(outputs) == 2:
                return outputs[0].squeeze(), outputs[1].squeeze()
            if len(outputs) == 1:
                o = outputs[0]
                if hasattr(o, "shape") and o.shape[-1] == 2:
                    return o[..., 0].squeeze(), o[..., 1].squeeze()

        # dict
        if isinstance(outputs, dict):
            keys = {k.lower(): k for k in outputs.keys()}
            v_key = next((keys[k] for k in ["valence","v","val"]), None)
            a_key = next((keys[k] for k in ["arousal","a","aro"]), None)
            if v_key is not None and a_key is not None:
                return outputs[v_key].squeeze(), outputs[a_key].squeeze()
            # single tensor in dict?
            if len(outputs) == 1:
                only = next(iter(outputs.values()))
                if hasattr(only, "shape") and only.shape[-1] == 2:
                    return only[...,0].squeeze(), only[...,1].squeeze()

        # single array
        if hasattr(outputs, "shape") and outputs.shape[-1] == 2:
            return outputs[..., 0].squeeze(), outputs[..., 1].squeeze()

        raise ValueError("Could not parse valence/arousal from model outputs.")

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
            preds_v, preds_a = self._predict_safe(epochs_window)
        except Exception as e:
            QMessageBox.critical(self, "Predict Error", f"Prediction failed:\n{e}")
            return

        # Aggregate
        avg_v = float(np.mean(preds_v))
        avg_a = float(np.mean(preds_a))

        val_label = "High Valence" if avg_v > 0.5 else "Low Valence"
        aro_label = "High Arousal" if avg_a > 0.5 else "Low Arousal"

        # Map to 4-quadrant demo emotions
        if val_label == "High Valence" and aro_label == "High Arousal":
            label = "Happy"
        elif val_label == "High Valence" and aro_label == "Low Arousal":
            label = "Calm"
        elif val_label == "Low Valence" and aro_label == "High Arousal":
            label = "Fear"
        else:
            label = "Sad"

        # Update visuals
        probs = [avg_v, avg_a, 1-avg_v, 1-avg_a]
        self.wheel.plot_wheel(highlight_label=label)
        self.bar.plot_confidences(["V_H","A_H","V_L","A_L"], probs)
        self.model_status.setText(
            f"Predicted: {label} | Valence: {avg_v:.2f}, Arousal: {avg_a:.2f}"
        )

        # Store last prediction for saving/logging
        self.last_prediction = {
            'label': label,
            'probs': {'V_H': float(avg_v), 'A_H': float(avg_a), 'V_L': float(1-avg_v), 'A_L': float(1-avg_a)},
            'window': {'start_epoch': int(start_idx), 'end_epoch_inclusive': int(end_idx-1)},
            'timestamp': datetime.now().isoformat()
        }

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    model_path = "1models_freeze_unfreeze_mvgt_bilstm/best_overall_mvgt_bilstm.h5"
    data_dir = "data/processed"
    app = QApplication(sys.argv)
    window = EEGDemoGUI(model_path, data_dir)
    window.show()
    sys.exit(app.exec())
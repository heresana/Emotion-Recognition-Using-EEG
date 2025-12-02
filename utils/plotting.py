import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os

# -------- Create output folder ----------
save_dir = "plots_output"
os.makedirs(save_dir, exist_ok=True)

# -------- Load DEAP subject file ----------
file_path = "data/raw/data_preprocessed_python/s01.dat"
with open(file_path, "rb") as f:
    data = pickle.load(f, encoding="latin1")

eeg = data['data']
fs = 128

# -------- Plot all channels ----------
trial = 0
trial_data = eeg[trial]

fig, axes = plt.subplots(8, 5, figsize=(22, 18))
axes = axes.flatten()

for i in range(40):
    axes[i].plot(trial_data[i])
    axes[i].set_title(f"Ch {i+1}", fontsize=8)
    axes[i].tick_params(labelsize=6)

plt.tight_layout()
plt.suptitle("EEG Time-Series: All 40 Channels (Trial 0)", fontsize=18, y=1.02)

save_path = os.path.join(save_dir, "eeg_channels.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()

print("Saved:", save_path)

fig, axes = plt.subplots(8, 5, figsize=(22, 18))
axes = axes.flatten()

#------------ALL 40 Spectrograms (8×5 grid)-----------

for i in range(40):
    freqs, times, Sxx = signal.spectrogram(trial_data[i], fs)
    axes[i].pcolormesh(times, freqs, Sxx, shading='gouraud')
    axes[i].set_title(f"Ch {i+1}", fontsize=8)
    axes[i].set_ylim(0, 60)
    axes[i].tick_params(labelsize=6)

plt.tight_layout()
plt.suptitle("EEG Spectrograms: All 40 Channels (Trial 0)", fontsize=18, y=1.02)

save_path = os.path.join(save_dir, "spectrograms.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()

print("Saved:", save_path)

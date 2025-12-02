import tensorflow as tf
from tensorflow.keras.layers import Input, Bidirectional, LSTM, Dropout, Dense, MultiHeadAttention, LayerNormalization, Add
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

# Model parameters
timesteps = 128  # example time steps (adjust as per your data)
channels = 32    # EEG channels
lstmunits = 64
numheads = 4
dropout_rate = 0.3

# Input layer
inputs = Input(shape=(timesteps, channels), name='Input')

# First BiLSTM layer
bilstm1 = Bidirectional(LSTM(lstmunits, return_sequences=True), name='BiLSTM1')(inputs)

# Dropout
drop1 = Dropout(dropout_rate, name='Dropout1')(bilstm1)

# Multi-head attention layer (fixed with value_dim=128 for residual match)
attn_output = MultiHeadAttention(num_heads=numheads, key_dim=lstmunits, value_dim=128, name='MultiHeadAttention')(drop1, drop1)

# Add & LayerNormalization (residual connection)
add = Add(name='Add')([attn_output, drop1])
norm = LayerNormalization(name='LayerNorm')(add)

# Second BiLSTM layer
bilstm2 = Bidirectional(LSTM(lstmunits), name='BiLSTM2')(norm)

# Dropout
drop2 = Dropout(dropout_rate, name='Dropout2')(bilstm2)

# Output dense layers for Valence and Arousal
valence_output = Dense(1, activation='sigmoid', name='Valence')(drop2)
arousal_output = Dense(1, activation='sigmoid', name='Arousal')(drop2)

# Model
model = Model(inputs=inputs, outputs=[valence_output, arousal_output], name='Baseline_BiLSTM_Model')

# Plot model architecture
tf.keras.utils.plot_model(model, to_file='baseline_bilstm_architecture.png', show_shapes=True, show_layer_names=True)

print(model.summary())
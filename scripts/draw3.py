import tensorflow as tf
from tensorflow.keras.layers import Input, Bidirectional, LSTM, Dropout, Dense, LayerNormalization, Add, Concatenate, GlobalAveragePooling1D
from tensorflow.keras.models import Model

# Custom SpatialEncoding and GraphMultiHeadAttention placeholder imports
# (replace these with your actual implementations if available)
from tensorflow.keras.layers import Layer

class SpatialEncoding(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def call(self, inputs):
        return inputs, tf.zeros_like(inputs)  # Dummy output and bias

class GraphMultiHeadAttention(Layer):
    def __init__(self, num_heads, key_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
    def call(self, inputs, bias=None):
        return inputs  # Dummy pass-through for visualization

# Model parameters
channels = 32
features = 17
d_model = 64
lstm_units = 64
dropout_rate = 0.3
num_heads = 2

# Input
inputs = Input(shape=(channels, features), name='Input')

# Spatial BiLSTM
spatial_lstm = Bidirectional(LSTM(lstm_units, return_sequences=True), name='Spatial_BiLSTM')(inputs)
drop1 = Dropout(dropout_rate, name='Dropout1')(spatial_lstm)
dense_proj = Dense(d_model, name='Dense_Projection')(drop1)

# Spatial Encoding Layer (outputs encoding + bias)
spatial_enc, graph_bias = SpatialEncoding(name='Spatial_Encoding')(dense_proj)

# Graph Multi-Head Attention Layer
g_attn = GraphMultiHeadAttention(num_heads=num_heads, key_dim=d_model, name='Graph_MultiHeadAttention')(spatial_enc, graph_bias)
add1 = Add(name='Add1')([g_attn, spatial_enc])
norm1 = LayerNormalization(name='LayerNorm1')(add1)

# Feedforward block (Dense + Add + Norm)
ff = Dense(d_model, activation='gelu', name='FeedForward')(norm1)
add2 = Add(name='Add2')([ff, norm1])
norm2 = LayerNormalization(name='LayerNorm2')(add2)

# Fusion BiLSTM (no return sequences)
fusion_lstm = Bidirectional(LSTM(lstm_units), name='Fusion_BiLSTM')(norm2)

# Global Average Pooling of spatial BiLSTM output
gap = GlobalAveragePooling1D(name='GlobalAvgPooling')(spatial_lstm)

# Concatenate fusion and pooled features
concat = Concatenate(name='Concatenate')([fusion_lstm, gap])

# Dense + Dropout
dense_ff = Dense(128, activation='relu', name='Dense_FF')(concat)
drop2 = Dropout(dropout_rate, name='Dropout2')(dense_ff)

# Output Layers
valence = Dense(1, activation='sigmoid', name='Valence')(drop2)
arousal = Dense(1, activation='sigmoid', name='Arousal')(drop2)

# Build model
model = Model(inputs=inputs, outputs=[valence, arousal], name='MVGT_BiLSTM_Model')

# Plot model architecture
tf.keras.utils.plot_model(model, to_file='mvgt_bilstm_architecture.png', show_shapes=True, show_layer_names=True)

print(model.summary())
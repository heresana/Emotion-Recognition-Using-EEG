import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Embedding, Add, LayerNormalization

CHANNEL_POS = np.random.uniform(-1, 1, (32, 3)).astype(np.float32)
CHANNEL_POS /= (np.linalg.norm(CHANNEL_POS, axis=1, keepdims=True) + 1e-6)

REGION_IDS = np.array([
    0,0,0,0,0,0,1,1,1,1,2,2,3,3,2,2,
    0,0,0,0,0,0,1,1,1,1,2,2,2,2,3,3
], dtype=np.int32)
K_GAUSSIAN = 32    # as in your training script

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

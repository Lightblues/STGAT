import tensorflow as tf
from tensorflow.python.keras.engine.base_layer import Layer


class DecoderLayer(Layer):
    def __init__(self, t_self_attention, s_self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu", have_enc=True):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.t_self_attention = t_self_attention
        self.s_self_attention = s_self_attention
        self.cross_attention = cross_attention
        self.conv1 = tf.keras.layers.Conv1D(filters=d_ff, kernel_size=1)
        self.conv2 = tf.keras.layers.Conv1D(filters=d_model, kernel_size=1)
        self.norm1_1 = tf.keras.layers.LayerNormalization()
        self.norm1_2 = tf.keras.layers.LayerNormalization()
        self.norm2 = tf.keras.layers.LayerNormalization()
        self.norm3 = tf.keras.layers.LayerNormalization()
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.activation = tf.keras.activations.elu if activation == "relu" else tf.keras.activations.gelu
        self.have_enc = have_enc

    def call(self, x, cross=None, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.t_self_attention(
           [x, x, x],
            attn_mask=x_mask
        ))
        x = self.norm1_1(x)

        x = x + self.dropout(self.s_self_attention(
           [x, x, x],
            attn_mask=x_mask
        ))
        x = self.norm1_2(x)


        if self.have_enc:
            x = x + self.dropout(self.cross_attention(
                [x, cross, cross],
                attn_mask=cross_mask
            ))

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y)))
        y = self.dropout(self.conv2(y))

#        return self.norm3(x+y)
        return x + self.norm3(y)


class Decoder(Layer):
    def __init__(self, layers, norm_layer=None):
        super(Decoder, self).__init__()
        self.layers = layers
        self.norm = norm_layer

    def call(self, x, cross=None, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)

        if self.norm is not None:
            x = self.norm(x)

        return x
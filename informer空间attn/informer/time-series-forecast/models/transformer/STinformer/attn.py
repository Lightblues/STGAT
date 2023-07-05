import tensorflow as tf
from tensorflow.python.keras.engine.base_layer import Layer
from math import sqrt
from utils.masking import TriangularCausalMask, ProbMask
import math
import numpy as np

class STFullAttention(Layer):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1):
        super(STFullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.dropout = tf.keras.layers.Dropout(attention_dropout)

    def call(self, inputs, attn_mask=None):
        queries, keys, values = inputs

        B, N, L, H, E = queries.shape
        B = tf.shape(queries)[0]
        _, M, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores_t = tf.einsum("bnlhe,bnshe->bnhls", queries, keys)
        A_t = self.dropout(tf.keras.activations.softmax(scale * scores_t, axis=-1))
        #        V = tf.einsum("bhls,bshd->bhld", A, values)
        V_t = tf.einsum("bnhls,bnshd->bnlhd", A_t, values)

        scores_s = tf.einsum("bnlhe,bmlhe->blhnm", queries, keys)
        # 加attention mask
        scores_s = tf.multiply(scores_s, tf.expand_dims(tf.expand_dims(attn_mask, axis=-3), axis=-4))
        A_s = self.dropout(tf.keras.activations.softmax(scale * scores_s, axis=-1))
        V = tf.einsum("blhnm,bmlhd->bnlhd", A_s, V_t)
        return V

class SFullAttention(Layer):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1):
        super(SFullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.dropout = tf.keras.layers.Dropout(attention_dropout)

    def call(self, inputs, attn_mask=None):
        queries, keys, values = inputs

        B, N, L, H, E = queries.shape
        B = tf.shape(queries)[0]
        _, M, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = tf.einsum("bnlhe,bmlhe->blhnm", queries, keys)
        if attn_mask is not None:
            # 假设原本是 0/1 mask, 转为 -inf/0 mask
            attn_mask = tf.cast(attn_mask, tf.float32)
            attn_mask = (1.0 - attn_mask) * -10000.0
            # add mask
            scores += attn_mask
            # scores = tf.multiply(scores, tf.expand_dims(tf.expand_dims(attn_mask, axis=-3), axis=-4))
        A = self.dropout(tf.keras.activations.softmax(scale * scores, axis=-1))
        V = tf.einsum("blhnm,bmlhd->bnlhd", A, values)

        return V


class FullAttention(Layer):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.dropout = tf.keras.layers.Dropout(attention_dropout)

    def call(self, inputs, attn_mask=None):
        queries, keys, values = inputs

        B, N, L, H, E = queries.shape
        B = tf.shape(queries)[0]
        _, _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)


        scores_t = tf.einsum("bnlhe,bnshe->bnhls", queries, keys)
        #if self.mask_flag:
        #    if attn_mask is None:
        #        attn_mask = TriangularCausalMask(B, L)
        #        # https://stackoverflow.com/questions/47447272/does-tensorflow-have-the-function-similar-to-pytorchs-masked-fill
        #        num = 3.4 * math.pow(10, 38)
        #        scores = (scores * attn_mask.mask) + (-((attn_mask.mask * num + num) - num))

        A = self.dropout(tf.keras.activations.softmax(scale * scores_t, axis=-1))
#        V = tf.einsum("bhls,bshd->bhld", A, values)
        V = tf.einsum("bnhls,bnshd->bnlhd", A, values)

        return V


class ProbAttention(Layer):
    def __init__(self, mask_flag=True, factor=1, scale=None, attention_dropout=0.1):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.dropout = tf.keras.layers.Dropout(attention_dropout)


    def _prob_QK(self, Q, K, sample_k, n_top):
        # Q [B, H, L, D]
        B, H, L, E = K.shape
        B = tf.shape(K)[0]
        _, _, S, _ = Q.shape
        S = tf.shape(Q)[2]

        # calculate the sampled Q_K
        #https://stackoverflow.com/questions/57716363/explicit-broadcasting-of-variable-batch-size-tensor
#        B,H,L,E = tf.where([True, True, True, True],
#                                   tf.shape(K), [0, 5, 5, 0])
#
#        K_expand = tf.broadcast_to(tf.expand_dims(K, -3), (B, H, S, L, E))
        K_expand = tf.tile(tf.expand_dims(K, -3), (1, 1, S, 1, 1))


#        indx_q_seq = tf.random.uniform((S,), maxval=L, dtype=tf.int32)
        indx_k_seq = tf.random.uniform((sample_k,), maxval=L, dtype=tf.int32)

#        K_sample = tf.gather(K_expand, tf.range(S), axis=2)
#        K_sample = tf.gather(K_sample, indx_q_seq, axis=2)
#        K_sample = tf.gather(K_sample, indx_k_seq, axis=3)
        K_sample = tf.gather(K_expand, indx_k_seq, axis=3)

        Q_K_sample = tf.squeeze(tf.matmul(tf.expand_dims(Q, -2), tf.einsum("...ij->...ji", K_sample)))
        # find the Top_k query with sparisty measurement
        M = tf.math.reduce_max(Q_K_sample, axis=-1) - tf.raw_ops.Div(x=tf.reduce_sum(Q_K_sample, axis=-1), y=L)
        M_top = tf.math.top_k(M, n_top, sorted=False)[1]
#        batch_indexes = tf.tile(tf.range(Q.shape[0])[:, tf.newaxis, tf.newaxis], (1, Q.shape[1], n_top))
        batch_indexes = tf.tile(tf.range(B)[:, tf.newaxis, tf.newaxis], (1, Q.shape[1], n_top))
#        head_indexes = tf.tile(tf.range(Q.shape[1])[tf.newaxis, :, tf.newaxis], (Q.shape[0], 1, n_top))
        head_indexes = tf.tile(tf.range(Q.shape[1])[tf.newaxis, :, tf.newaxis], (B, 1, n_top))
        idx = tf.stack(values=[batch_indexes, head_indexes, M_top], axis=-1)

        # use the reduced Q to calculate Q_K
        Q_reduce = tf.gather_nd(Q, idx)

        Q_K = tf.matmul(Q_reduce, tf.transpose(K, [0, 1, 3, 2]))

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        B = tf.shape(V)[0]
#        if not self.mask_flag:
#            V_sum = tf.reduce_sum(V, -2)
##            contex = tf.identity(tf.broadcast_to(tf.expand_dims(V_sum, -2), [B, H, L_Q, V_sum.shape[-1]]))
#            contex = tf.identity(tf.tile(tf.expand_dims(V_sum, -2), [1, 1, L_Q, 1]))
#        else:  # use mask
#            assert (L_Q == L_V)  # requires that L_Q == L_V, i.e. for self-attention only
#            contex = tf.math.cumsum(V, axis=-1)

        V_sum = tf.reduce_sum(V, -2)
        contex = tf.identity(tf.tile(tf.expand_dims(V_sum, -2), [1, 1, L_Q, 1]))
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q):
        B, H, L_V, D = V.shape
        B = tf.shape(V)[0]
#        if self.mask_flag:
#            attn_mask = ProbMask(B, H, L_Q, index, scores)
#
#            # scores.masked_fill_(attn_mask.mask, -np.inf)
#            num = 3.4 * math.pow(10, 38)
#            scores = (scores * attn_mask.mask) + (-((attn_mask.mask * num + num) - num))

        attn = tf.keras.activations.softmax(scores, axis=-1)  # nn.Softmax(dim=-1)(scores)
#        batch_indexes = tf.tile(tf.range(V.shape[0])[:, tf.newaxis, tf.newaxis], (1, V.shape[1], index.shape[-1]))
#        head_indexes = tf.tile(tf.range(V.shape[1])[tf.newaxis, :, tf.newaxis], (V.shape[0], 1, index.shape[-1]))

        batch_indexes = tf.tile(tf.range(B)[:, tf.newaxis, tf.newaxis], (1, V.shape[1], scores.shape[2]))
        head_indexes = tf.tile(tf.range(V.shape[1])[tf.newaxis, :, tf.newaxis], (B, 1, scores.shape[2]))
        idx = tf.stack(values=[batch_indexes, head_indexes, index], axis=-1)

        context_in = tf.tensor_scatter_nd_update(context_in, idx, tf.matmul(attn, V))

        return tf.convert_to_tensor(context_in)

    @tf.function
    def call(self, inputs, attn_mask=None):
        queries, keys, values = inputs
        B, L, H, D = queries.shape
        B = tf.shape(queries)[0]
        _, S, _, _ = keys.shape

#        queries = tf.reshape(queries, (B, H, L, -1))
#        keys = tf.reshape(keys, (B, H, S, -1))
#        values = tf.reshape(values, (B, H, S, -1))

        queries = tf.reshape(queries, (B, H, L, 64))
        keys = tf.reshape(keys, (B, H, S, 64))
        values = tf.reshape(values, (B, H, S, 64))

        U = self.factor * np.ceil(np.log(S)).astype('int').item()
        u = self.factor * np.ceil(np.log(L)).astype('int').item()

        scores_top, index = self._prob_QK(queries, keys, u, U)
        # add scale factor
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L)
        # update the context with selected top_k queries
        context = self._update_context(context, values, scores_top, index, L)

        return context


class AttentionLayer(Layer):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)
        self.d_model = d_model

        self.inner_attention = attention

        self.query_projection = tf.keras.layers.Dense(d_keys * n_heads)
        self.key_projection = tf.keras.layers.Dense(d_keys * n_heads)
        self.value_projection = tf.keras.layers.Dense(d_values * n_heads)
        self.query_norm = tf.keras.layers.LayerNormalization()
        self.key_norm = tf.keras.layers.LayerNormalization()
        self.value_norm = tf.keras.layers.LayerNormalization()

#        self.query_projection_t = tf.keras.layers.Dense(d_keys * n_heads)
#        self.key_projection_t = tf.keras.layers.Dense(d_keys * n_heads)
#        self.value_projection_t = tf.keras.layers.Dense(d_values * n_heads)
#        self.query_norm_t = tf.keras.layers.LayerNormalization()
#        self.key_norm_t = tf.keras.layers.LayerNormalization()
#        self.value_norm_t = tf.keras.layers.LayerNormalization()
#
#        self.query_projection_s = tf.keras.layers.Dense(d_keys * n_heads)
#        self.key_projection_s = tf.keras.layers.Dense(d_keys * n_heads)
#        self.value_projection_s = tf.keras.layers.Dense(d_values * n_heads)
#        self.query_norm_s = tf.keras.layers.LayerNormalization()
#        self.key_norm_s = tf.keras.layers.LayerNormalization()
#        self.value_norm_s = tf.keras.layers.LayerNormalization()

        self.out_projection = tf.keras.layers.Dense(d_model)
        self.out_norm = tf.keras.layers.LayerNormalization()

        self.n_heads = n_heads

#    def build(self, input_shape):
#        print(input_shape)
#        #B-batch;
#        B, L, _ = input_shape[0]
#        _, S, _ = input_shape[1]
#        H = self.n_heads
#
#
#        self.queries = self.add_weight(shape=(B, L, H, self.d_model),
#                                 initializer='random_normal',
#                                 trainable=True,
#                                 name='queryies')
#
#        self.keys = self.add_weight(shape=(B, S, H, self.d_model),
#                                       initializer='random_normal',
#                                       trainable=True,
#                                       name='keys')
#
#        self.values = self.add_weight(shape=(B, S, H, self.d_model),
#                                       initializer='random_normal',
#                                       trainable=True,
#                                       name='values')
    @tf.function
    def call(self, inputs, attn_mask=None):
        queries, keys, values = inputs
        B, N, L, _ = queries.shape
        _, _, S, _ = keys.shape
        H = self.n_heads

#        self.queries = tf.reshape(self.query_projection(queries), (B, L, H, -1))
#        self.keys = tf.reshape(self.key_projection(keys), (B, S, H, -1))
#        self.values = tf.reshape(self.value_projection(values), (B, S, H, -1))
#
#        out = tf.reshape(self.inner_attention([self.queries, self.keys, self.values], attn_mask=attn_mask), (B, L, -1))

#        queries = tf.stack(tf.split(queries, H, axis=2), axis=2)
#        keys = tf.stack(tf.split(keys, H, axis=2), axis=2)
#        values = tf.stack(tf.split(values, H, axis=2), axis=2)

        queries = tf.stack(tf.split(self.query_norm(self.query_projection(queries)), H, axis=3), axis=3)
        keys = tf.stack(tf.split(self.key_norm(self.key_projection(keys)), H, axis=3), axis=3)
        values = tf.stack(tf.split(self.value_norm(self.value_projection(values)), H, axis=3), axis=3)

        out = self.inner_attention([queries, keys, values], attn_mask=attn_mask)
        out = tf.squeeze(tf.concat(tf.split(out, H, axis=3), axis=4), axis=3)

#        return out
        return self.out_norm(self.out_projection(out))


if __name__ == '__main__':
        attn = AttentionLayer(FullAttention(False), 128, 4)
        queries = tf.zeros((32, 20, 128))
        keys = tf.zeros((32, 20, 128))
        values = tf.zeros((32, 20, 128))
        print(attn([queries, keys, values]))

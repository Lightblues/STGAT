import tensorflow as tf
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.keras.layers import LeakyReLU
import math
import numpy as np

# 位置embedding, 偶数sin奇数cos
class PositionalEmbedding(Layer):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = np.zeros((max_len, d_model), dtype=np.float32)

        position = np.expand_dims(np.arange(0, max_len, dtype=np.float32), 1)
        div_term = np.exp(np.arange(0, d_model, 2, dtype=np.float32) * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)

        self.pe = tf.expand_dims(tf.convert_to_tensor(pe), 0)

    def call(self, inputs, **kwargs):
        return self.pe[:, :inputs.shape[1]]

# 数据embedding, 对原始数据进行1维卷积
class TokenEmbedding(Layer):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        self.tokenConv = tf.keras.layers.Conv1D(filters=d_model,
                                    kernel_size=3, padding='causal', activation='linear')
        self.activation = LeakyReLU()

    def call(self, inputs, **kwargs):
        x = self.tokenConv(inputs)
        x = self.activation(x)

        return x


class FixedEmbedding(Layer):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = np.zeros((c_in, d_model), dtype=np.float32)

        position = np.expand_dims(np.arange(0, c_in, dtype=np.float32), 1)
        div_term = np.exp(np.arange(0, d_model, 2, dtype=np.float32) * -(math.log(10000.0) / d_model))

        w[:, 0::2] = np.sin(position * div_term)
        w[:, 1::2] = np.cos(position * div_term)

        w = tf.convert_to_tensor(w)
        tf.stop_gradient(w)
        w = tf.keras.initializers.Constant(w)
        self.emb = tf.keras.layers.Embedding(c_in, d_model, embeddings_initializer=w)

    def call(self, inputs, **kargs):
        embedding = self.emb(inputs)

        return embedding

# 时间embedding 
class TemporalEmbedding(Layer):
    def __init__(self, d_model, embed_type='fixed', data='ETTh'):
        super(TemporalEmbedding, self).__init__()

        embed_type = 'learned'
        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13
        date_type_size = 5
        holiday_type_size = 8
        week_of_year_size = 54
        id_size = 101

        #fixedembedding:使用位置编码作为embedding的参数, 无需训练参数
        Embed = FixedEmbedding if embed_type == 'fixed' else tf.keras.layers.Embedding
        if data == 'ETTm':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

        self.date_type_embed = Embed(date_type_size, d_model)
        self.holiday_type_embed = Embed(holiday_type_size, d_model)
        self.week_of_year_embed = Embed(week_of_year_size, d_model)
        self.id_embed = Embed(id_size, d_model)

    def call(self, inputs, **kargs):
        # x = x.long()

        id_x = self.id_embed(inputs[:, :, 6])
        week_of_year_x = self.week_of_year_embed(inputs[:, :, 5])
        holiday_type_x = self.holiday_type_embed(inputs[:, :, 4])
        date_type_x = self.date_type_embed(inputs[:, :, 3])

        weekday_x = self.weekday_embed(inputs[:, :, 2])
        day_x = self.day_embed(inputs[:, :, 1])
        month_x = self.month_embed(inputs[:, :, 0])

        return id_x + week_of_year_x + holiday_type_x + date_type_x + weekday_x + day_x + month_x


class DataEmbedding(Layer):
    def __init__(self, c_in, d_model, embed_type='fixed', data='ETTh', dropout=0.1, seq_len=96):
        super(DataEmbedding, self).__init__()
        self.c_in = c_in
        self.seq_len = seq_len

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        ##位置特征
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        ##时间特征
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, data=data)

        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, x, x_mark=None, **kwargs):

        x = self.value_embedding(x) + self.position_embedding(x) + self.temporal_embedding(x_mark)

        return self.dropout(x)

class DataEmbedding_v2(Layer):
    def __init__(self, c_in, d_model, embed_type='fixed', data='ETTh', dropout=0.1, seq_len=96):
        super(DataEmbedding, self).__init__()

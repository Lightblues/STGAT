import logging
import os
import sys
# sys.path.append("D:/work/code_repository/vacation-ai/time-series-forecast")
current_path = os.path.dirname(__file__)
parent_path = os.path.dirname(current_path)
out_path = os.path.dirname(parent_path)
sys.path.append(parent_path)
import tensorflow as tf
from models.transformer.informer.informer import Informer
from models.transformer.STinformer.STinformer import STInformer

# model = STInformer(7, 7, 7, 96, 48, 24, 32)
# x_enc = tf.zeros((32, 96, 7))
# x_dec = tf.zeros((32, 72, 7))
# x_mark_enc = tf.zeros((32, 96, 4))
# x_mark_dec = tf.zeros((32, 72, 4))
# print(model([x_enc, x_dec, x_mark_enc, x_mark_dec]).shape)

model = STInformer(enc_in=7, dec_in=7, c_out=7, seq_len=96, label_len=48, out_len=24, out_len_start=1, batch_size=32)
# 注意 STInformer 不包含embedding
d_model = 512
x_enc = tf.zeros((32, 10, 96, d_model))
x_dec = tf.zeros((32, 10, 72, d_model))
# x_mark_enc = tf.zeros((32, 10, 96, 4))
# x_mark_dec = tf.zeros((32, 10, 72, 4))
print(model([x_enc, x_dec]).shape)
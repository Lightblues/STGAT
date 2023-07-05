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

model = Informer(7, 7, 7, 96, 48, 24, 32)
x_enc = tf.zeros((32, 96, 7))
x_dec = tf.zeros((32, 72, 7))
x_mark_enc = tf.zeros((32, 96, 4))
x_mark_dec = tf.zeros((32, 72, 4))
print(model([x_enc, x_dec]).shape)

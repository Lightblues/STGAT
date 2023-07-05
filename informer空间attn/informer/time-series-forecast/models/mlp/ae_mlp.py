"""
@Time : 2023/5/9 17:34
@Author : mcxing
@File : ae_mlp.py
@Software: PyCharm
"""

import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler, KBinsDiscretizer
import json
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras.layers import Embedding, Concatenate, Flatten, Dense, Input, BatchNormalization, Activation, \
    Dropout, LSTM, GRU, Concatenate, LayerNormalization
from functools import reduce
from models.base.base_model import BaseModel
import logging
import sys

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stderr, level=logging.INFO, format=
'''[%(levelname)s] [%(asctime)s] [%(threadName)s] [%(name)s] '''
'''[%(filename)s:%(funcName)s:%(lineno)d]: %(message)s''')


class AEMLPModel(BaseModel):
    def __init__(self, args):
        super(AEMLPModel, self).__init__(args)

        self.custom_params = json.loads(self.args.custom_params) if self.args.custom_params else {}
        self.ae = self.custom_params['ae'] if 'ae' in self.custom_params.keys() else False
        self.embedding_size = self.custom_params[
            'embedding_size'] if 'embedding_size' in self.custom_params.keys() else 8
        self.feature_bins = self.custom_params[
            'feature_bins'] if 'feature_bins' in self.custom_params.keys() else 512
        self.discretizer_strategy = self.custom_params[
            'discretizer_strategy'] if 'discretizer_strategy' in self.custom_params.keys() else ['uniform',
                                                                                                 'quantile']

        self.model_params['learning_rate'] = self.custom_params[
            'learning_rate'] if 'learning_rate' in self.custom_params.keys() else 0.01
        self.model_params['l2_reg'] = self.custom_params['l2_reg'] if 'l2_reg' in self.custom_params.keys() else 0.2
        self.model_params['batch_size'] = self.custom_params[
            'batch_size'] if 'batch_size' in self.custom_params.keys() else 256

    def create_model(self):

        self.history = [[None for i in range(self.args.forecast_days)] for j in range(len(self.rank_bins) - 1)]
        self.discretizers = [[[] for i in range(self.args.forecast_days)] for j in range(len(self.rank_bins) - 1)]

        return [[] for i in range(len(self.rank_bins) - 1)]

    def create_ae_mlp(self, num_columns, l2_reg, lr):
        hidden_units_encoder = [256, 64]
        hidden_units_decoder = [256, 64]
        hidden_units_ae = [512, 512, 256]
        hidden_units_out = [512, 256, 64]
        stddev = 0.03
        dropout_rate = 0.1

        inp = Input(shape=(num_columns,))
        if self.ae:
            encoder = tf.keras.layers.GaussianNoise(stddev)(x0)
            for unit in hidden_units_encoder:
                encoder = Dense(unit, kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(encoder)
                encoder = BatchNormalization()(encoder)
                encoder = Activation('relu')(encoder)
                encoder = Dropout(dropout_rate)(encoder)

            decoder = encoder
            for unit in hidden_units_decoder:
                decoder = Dense(unit, kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(decoder)
                decoder = BatchNormalization()(decoder)
                decoder = Activation('relu')(decoder)
                decoder = Dropout(dropout_rate)(decoder)
            decoder = Dense(num_columns, name='decoder')(decoder)

            # decoder输出做预测
            x_ae = decoder
            for unit in hidden_units_ae:
                x_ae = Dense(unit, kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x_ae)
                x_ae = BatchNormalization()(x_ae)
                x_ae = Activation('relu')(x_ae)
                x_ae = Dropout(dropout_rate)(x_ae)
            out_ae = Dense(1, activation=None, name='ae_action')(x_ae)

            # final output
            x = Concatenate()([inp, encoder])
            # x = tf.keras.layers.BatchNormalization()(x)
            # x = tf.keras.layers.Dropout(dropout_rates[3])(x)

            for unit in hidden_units_out:
                x = Dense(unit, kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)
                x = Dropout(dropout_rate)(x)

            out = Dense(1, activation=None, name='action')(x)

            model = tf.keras.models.Model(inputs=inp, outputs=[decoder, out_ae, out])
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                          loss={'decoder': tf.keras.losses.MeanSquaredError(),
                                'ae_action': tf.keras.losses.MeanSquaredError(),
                                'action': tf.keras.losses.MeanSquaredError(),
                                },
                          loss_weights={'decoder': 1000,
                                        'ae_action': 1,
                                        'action': 1
                                        },
                          metrics=['MSE', 'MAE', 'MAPE']
                          )
        else:
            x = Flatten()(Concatenate(axis=1)([Embedding(self.feature_bins, self.embedding_size,
                                                         embeddings_regularizer=tf.keras.regularizers.l2(l2_reg))(
                inp[:, i]) for i in range(num_columns)]))

            #            x = Flatten()(Embedding(self.feature_bins, self.embedding_size, input_length=num_columns,embeddings_regularizer=tf.keras.regularizers.l2(l2_reg))(inp))

            for unit in hidden_units_out:
                x = Dense(unit, kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)
                x = Dropout(dropout_rate)(x)
            if self.args.output_type == 'multi_output':
                out = Dense(self.args.forecast_days, activation=None, name='action')(x)
            else:
                out = Dense(1, activation=None, name='action')(x)

            model = tf.keras.models.Model(inputs=inp, outputs=[out])
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                          loss={
                              'action': tf.keras.losses.MeanSquaredError(),
                          },
                          metrics=['MSE', 'MAE', 'MAPE']
                          )

            return model

    def model_tune_builder(self, hp):
        hp_l2_reg = hp.Choice('l2_reg', values=[0.01, 0.05, 0.1, 0.2, 0.5])
        hp_lr = hp.Choice('learning_rate', values=[0.001, 0.005, 0.01, 0.02])
        model = self.create_ae_mlp(num_columns=self.x_num, l2_reg=hp_l2_reg, lr=hp_lr)
        return model

    def tune_model(self, train_x, train_y, train_w, val_x=None, val_y=None, val_w=None):
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0.0001, verbose=1)
        earlystopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=6, verbose=1, mode='min',
                                      restore_best_weights=True)

        tuner = kt.RandomSearch(self.model_tune_builder,
                                objective='val_MSE',
                                max_trials=20,
                                seed=2022,
                                directory='my_dir',
                                project_name='randomsearch',
                                overwrite=True
                                )

        if self.ae:
            tuner.search(train_x, [train_x, train_y, train_y]
                         , epochs=100
                         , batch_size=256
                         , verbose=1
                         , sample_weight=[train_w, train_w, train_w]
                         , validation_data=(val_x, [val_x, val_y, val_y], val_w)
                         , callbacks=[reduce_lr, earlystopping]
                         )
        else:
            tuner.search(train_x, train_y
                         , epochs=100
                         , batch_size=256
                         , verbose=3
                         , sample_weight=train_w
                         , validation_data=(val_x, val_y, val_w)
                         , callbacks=[reduce_lr, earlystopping]
                         )
        best_hps = tuner.get_best_hyperparameters()[0]
        self.model_params['learning_rate'] = best_hps.get('learning_rate')
        self.model_params['l2_reg'] = best_hps.get('l2_reg')

    def fit_model(self, _bin, _day, train_x, train_y, train_w, val_x, val_y, val_w):

        if len(self.discretizer_strategy) > 0:
            if 'uniform' in self.discretizer_strategy:
                self.discretizers[_bin][_day].append(
                    KBinsDiscretizer(n_bins=self.feature_bins, encode='ordinal', strategy='uniform'))
            if 'quantile' in self.discretizer_strategy:
                self.discretizers[_bin][_day].append(
                    KBinsDiscretizer(n_bins=self.feature_bins, encode='ordinal', strategy='quantile'))
            train_x = reduce(lambda x1, x2: np.append(x1, values=x2, axis=1),
                             list(map(lambda f: f.fit_transform(train_x), self.discretizers[_bin][_day])))
            val_x = reduce(lambda x1, x2: np.append(x1, values=x2, axis=1),
                           list(map(lambda f: f.transform(val_x), self.discretizers[_bin][_day])))

        self.x_num = train_x.shape[1]
        logger.info(">>>>>>>>>>{}th bin {}th model params: {}".format(_bin + 1, _day + 1, self.model_params))
        self.models[_bin].append(self.create_ae_mlp(num_columns=self.x_num, l2_reg=self.model_params['l2_reg'],
                                                    lr=self.model_params['learning_rate']))

        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.0001, verbose=1)
        earlystopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=8, verbose=1, mode='min',
                                      restore_best_weights=True)
        if self.ae:
            self.history[_bin][_day] = self.models[_bin][_day].fit(train_x, [train_x, train_y, train_y]
                                                                   , epochs=100
                                                                   , batch_size=self.model_params['batch_size']
                                                                   , verbose=2
                                                                   , sample_weight=[train_w, train_w, train_w]
                                                                   ,
                                                                   validation_data=(val_x, [val_x, val_y, val_y], val_w)
                                                                   , callbacks=[reduce_lr, earlystopping]
                                                                   )
        #            self.model_diagrams[_bin][_day].fit(val_x, [val_x, val_y, val_y]
        #                                        , epochs=len(self.history[_bin][_day].epoch) - 6
        #                                        , batch_size=256
        #                                        , verbose=2
        #                                        , sample_weight=[val_w, val_w, val_w]
        #                                        )
        else:
            #            self.history[_bin][_day] = self.model_diagrams[_bin][_day].fit(train_x
            #                                                                   , train_y
            #                                                                   , epochs=100
            #                                                                   , batch_size=self.model_params['batch_size']
            #                                                                   , verbose=2
            #                                                                   , sample_weight=train_w
            #                                                                   , validation_data=(val_x, val_y, val_w)
            #                                                                   , callbacks=[reduce_lr, earlystopping]
            #                                                                   )

            self.history[_bin][_day] = self.models[_bin][_day].fit(np.append(train_x, values=val_x, axis=0)
                                                                   , np.append(train_y, values=val_y, axis=0)
                                                                   , epochs=100
                                                                   , batch_size=self.model_params['batch_size']
                                                                   , verbose=2
                                                                   , sample_weight=np.append(train_w, values=val_w,
                                                                                             axis=0)
                                                                   , validation_data=(val_x, val_y, val_w)
                                                                   , callbacks=[reduce_lr, earlystopping]
                                                                   )

    #
    def predict_model(self, _bin, _day, predict_x):
        predict_x = reduce(lambda x1, x2: np.append(x1, values=x2, axis=1),
                           list(map(lambda f: f.transform(predict_x), self.discretizers[_bin][_day])))
        if self.ae:
            return self.models[_bin][_day].predict(predict_x)[1].reshape(-1).tolist()
        else:  #
            return self.models[_bin][_day].predict(predict_x).reshape(-1).tolist()

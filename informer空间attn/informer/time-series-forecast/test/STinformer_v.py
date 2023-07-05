"""
@Time : 2023/1/1 22:18
@Author : mcxing
@File : STinformer.py
@Software: PyCharm
"""

import argparse
import numpy as np
import pandas as pd
import math
import datetime
from sklearn.preprocessing import StandardScaler
import json
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras.layers import Embedding, Concatenate, Flatten, Dense, Input, BatchNormalization, Activation, \
    Dropout, LSTM, GRU, Concatenate, LayerNormalization
from models.base.base_model import BaseModel
from utils.data_processing import DataGenerater
import logging
import sys
import os
import multiprocessing
from models.transformer.informer.embed import DataEmbedding
from models.transformer.STinformer.attn import ProbAttention, FullAttention, SFullAttention, STFullAttention, AttentionLayer
from models.transformer.STinformer.encoder import ConvLayer, Encoder, EncoderLayer
from models.transformer.STinformer.decoder import Decoder, DecoderLayer

os.environ["PYSPARK_PYTHON"] = "/usr/bin/python3.6.5"

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stderr, level=logging.INFO, format=
'''[%(levelname)s] [%(asctime)s] [%(threadName)s] [%(name)s] '''
'''[%(filename)s:%(funcName)s:%(lineno)d]: %(message)s''')


class STInformerModel(BaseModel):
    def __init__(self, args):
        super(STInformerModel, self).__init__(args)

        self.custom_params = json.loads(self.args.custom_params) if self.args.custom_params else {}
        self.train_date_gap = self.custom_params[
            'train_date_gap'] if 'train_date_gap' in self.custom_params.keys() else 1
        self.model_params['embedding_size'] = self.custom_params[
            'embedding_size'] if 'embedding_size' in self.custom_params.keys() else 32
        self.model_params['learning_rate'] = self.custom_params[
            'learning_rate'] if 'learning_rate' in self.custom_params.keys() else 0.1
        self.model_params['l2_reg'] = self.custom_params['l2_reg'] if 'l2_reg' in self.custom_params.keys() else 0
        self.model_params['dropout'] = self.custom_params['dropout'] if 'dropout' in self.custom_params.keys() else 0.2
        self.model_params['batch_size'] = self.custom_params[
            'batch_size'] if 'batch_size' in self.custom_params.keys() else 256
        self.model_params['latent_dim'] = self.custom_params[
            'latent_dim'] if 'latent_dim' in self.custom_params.keys() else 512
        self.model_params['dense_dims'] = self.custom_params[
            'dense_dims'] if 'dense_dims' in self.custom_params.keys() else [512, 256, 64, 8]
        self.model_params['activation'] = self.custom_params[
            'activation'] if 'activation' in self.custom_params.keys() else 'gelu'

        self.model_params['enc_in'] = self.custom_params[
            'enc_in'] if 'enc_in' in self.custom_params.keys() else 512
        self.model_params['dec_in'] = self.custom_params[
            'dec_in'] if 'dec_in' in self.custom_params.keys() else 512
        self.model_params['d_model'] = self.custom_params[
            'd_model'] if 'd_model' in self.custom_params.keys() else 512
        self.model_params['d_ff'] = self.custom_params[
            'd_ff'] if 'd_ff' in self.custom_params.keys() else 512
        self.model_params['c_out'] = self.custom_params[
            'c_out'] if 'c_out' in self.custom_params.keys() else 1
        self.model_params['n_heads'] = self.custom_params[
            'n_heads'] if 'n_heads' in self.custom_params.keys() else 8
        self.model_params['e_layers'] = self.custom_params[
            'e_layers'] if 'e_layers' in self.custom_params.keys() else 1
        self.model_params['d_layers'] = self.custom_params[
            'd_layers'] if 'd_layers' in self.custom_params.keys() else 1
        self.model_params['factor'] = self.custom_params[
            'factor'] if 'factor' in self.custom_params.keys() else 1

        self.model_params['loss'] = tf.keras.losses.MeanSquaredError()
        if 'loss' in self.custom_params.keys():
            if self.custom_params['loss'] == 'msle':
                self.model_params['loss'] = tf.keras.losses.MeanSquaredLogarithmicError()


        self.model_params['attn_type'] = self.custom_params['attn_type'] if 'attn_type' in self.custom_params.keys() else 'full'

        self.encoder_timesteps = self.custom_params[
            'encoder_timesteps'] if 'encoder_timesteps' in self.custom_params.keys() else 30
        self.token_timesteps = self.custom_params[
            'token_timesteps'] if 'token_timesteps' in self.custom_params.keys() else 15
        self.decoder_timesteps = self.args.forecast_days
        self.decoder_timesteps_start = self.args.forecast_days_start
        self.validation_days = self.args.validation_days
        attn_mask = [[0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
        0., 0., 0., 1., 1., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
       [0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
       [0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
       [1., 0., 1., 0., 0., 0., 1., 0., 1., 1., 1., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 1., 1., 0., 0., 0., 0.,
        1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 1., 0.,
        1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 1., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
       [0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 1., 0., 1., 1., 0., 0.,
        0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0.,
        0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 1., 1., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
       [0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1.,
        1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
        1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 1., 1.,
        0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
        1., 0., 1., 1., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
        0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0.,
        1., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 1., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0.],
       [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 1., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 1., 1., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 1., 0., 1., 1., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 1., 0., 1., 0., 1., 0., 0., 0., 1., 0.],
       [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 1., 1., 0., 1., 1., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
        0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 1., 0.],
       [0., 0., 0., 0., 1., 1., 1., 1., 0., 0., 1., 1., 1., 1., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 1., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.]]
        self.attn_mask =  tf.convert_to_tensor(attn_mask)


    def train_model(self):
        logger.info(">>>>>>>>>>Start training model:")
        logger.info(">>>>>>>>>>model_type:{}, output_type:{}".format(self.args.model_type, self.args.output_type))

        if self.dataset is None:
            raise Exception(">>>>>>>>>>There is no data!")
        ## data preprocess
        self.encoder_dense_feature_cols = self.args.encoder_dense_feature_cols
        self.encoder_sparse_feature_cols = self.args.encoder_sparse_feature_cols
        self.decoder_dense_feature_cols = self.args.decoder_dense_feature_cols
        self.decoder_sparse_feature_cols = self.args.decoder_sparse_feature_cols
        self.decoder_output_col = ['value_real']

        self.all_dense_feature_cols = list(set(self.encoder_dense_feature_cols + self.decoder_dense_feature_cols))
        self.all_sparse_feature_cols = list(set(self.encoder_sparse_feature_cols + self.decoder_sparse_feature_cols))

        self.encoder_sparse_feature_index = list(
            map(lambda x: self.all_sparse_feature_cols.index(x), self.encoder_sparse_feature_cols))
        self.decoder_sparse_feature_index = list(
            map(lambda x: self.all_sparse_feature_cols.index(x), self.decoder_sparse_feature_cols))

        end_date = (datetime.datetime.strptime(self.args.cur_date, "%Y-%m-%d") + datetime.timedelta(
            days=(self.args.forecast_days - 1))).strftime("%Y-%m-%d")
        self.dataset = self.dataset[self.dataset['date'] <= end_date]
        self.dataset['value_real'] = self.dataset['value']
        self.all_sparse_cols_size = self.dataset[self.all_sparse_feature_cols].max().values.astype(np.int32) + 1
        scaler = StandardScaler()
        self.dataset[self.all_dense_feature_cols] = scaler.fit_transform(self.dataset[self.all_dense_feature_cols])

        self.models = self.create_model()
        merged_res_df = pd.DataFrame()
        for _bin in range(len(self.rank_bins) - 1):
            bin_dataset = self.dataset[
                (self.dataset['rank'] > self.rank_bins[_bin]) & (
                        self.dataset['rank'] <= self.rank_bins[_bin + 1])].dropna()
            logger.info(
                ">>>>>>>>>>{}th bin {} starts".format(_bin + 1, [self.rank_bins[_bin], self.rank_bins[_bin + 1]]))

            ## 去掉疫情高峰期           
            if self.args.drop_covid19 == 1:
                bin_dataset = bin_dataset.query(
                    "not(date>='2020-01-01' and date<='2020-05-31') and not(date>='2021-08-01' and date<='2021-08-31') and not(date>='2022-03-01' and date<='2022-05-31')")
            ## 去掉2020,2021,2022疫情三年
            elif self.args.drop_covid19 == 2:
                bin_dataset = bin_dataset.query(
                    "not(date>='2020-01-01' and date<='2022-11-30')")
            ## 去掉2020,2022两年                   
            elif self.args.drop_covid19 == 3:
                bin_dataset = bin_dataset.query(
                    "not(date>='2020-01-01' and date<='2022-11-30') or (date>='2021-01-01' and date<='2021-12-31')")

            pr = multiprocessing.Pool(8)
            prs = []
            for id, sdf in bin_dataset.groupby('id'):
                prs.append(pr.apply_async(DataGenerater.encoder_decoder_data_generater, args=(
                    id, sdf, self.validation_days, self.encoder_timesteps, self.decoder_timesteps,
                    self.encoder_dense_feature_cols, self.encoder_sparse_feature_cols,
                    self.decoder_dense_feature_cols, self.decoder_sparse_feature_cols,
                    self.decoder_output_col, self.train_date_gap, self.args.differential_col,
                    self.custom_params["start_token_len"], self.decoder_timesteps_start)))
            pr.close()
            pr.join()
            train_encoder_dense_input_data = []
            train_encoder_sparse_input_data = []
            train_decoder_dense_input_data = []
            train_decoder_sparse_input_data = []
            train_output_data = []
            train_weight_data = []
            train_encoder_pos_data = []
            train_decoder_pos_data = []

            val_encoder_dense_input_data = []
            val_encoder_sparse_input_data = []
            val_decoder_dense_input_data = []
            val_decoder_sparse_input_data = []
            val_output_data = []
            val_weight_data = []
            val_encoder_pos_data = []
            val_decoder_pos_data = []

            predict_encoder_dense_input_data = []
            predict_encoder_sparse_input_data = []
            predict_decoder_dense_input_data = []
            predict_decoder_sparse_input_data = []
            predict_encoder_pos_data = []
            predict_decoder_pos_data = []

            for x in prs:
                data = x.get()
                train_encoder_dense_input_data.append(data[0])
                train_encoder_sparse_input_data.append(data[1])
                train_decoder_dense_input_data.append(data[2])
                train_decoder_sparse_input_data.append(data[3])
                train_output_data.append(data[4])
                train_weight_data.append(data[5])
                train_encoder_pos_data.append(data[6])
                train_decoder_pos_data.append(data[7])
                val_encoder_dense_input_data.append(data[8])
                val_encoder_sparse_input_data.append(data[9])
                val_decoder_dense_input_data.append(data[10])
                val_decoder_sparse_input_data.append(data[11])
                val_output_data.append(data[12])
                val_weight_data.append(data[13])
                val_encoder_pos_data.append(data[14])
                val_decoder_pos_data.append(data[15])
                predict_encoder_dense_input_data.append(data[16])
                predict_encoder_sparse_input_data.append(data[17])
                predict_encoder_pos_data.append(data[18])
                predict_decoder_dense_input_data.append(data[19])
                predict_decoder_sparse_input_data.append(data[20])
                predict_decoder_pos_data.append(data[21])

            train_encoder_dense_input_data = np.array(list(map(list, zip(*train_encoder_dense_input_data))))
            train_encoder_sparse_input_data = np.array(list(map(list, zip(*train_encoder_sparse_input_data))))
            train_decoder_dense_input_data = np.array(list(map(list, zip(*train_decoder_dense_input_data))))
            train_decoder_sparse_input_data = np.array(list(map(list, zip(*train_decoder_sparse_input_data))))
            train_output_data = np.array(list(map(list, zip(*train_output_data))))
            train_weight_data = np.array(list(map(list, zip(*train_weight_data))))
            train_encoder_pos_data = np.array(list(map(list, zip(*train_encoder_pos_data))))
            train_decoder_pos_data = np.array(list(map(list, zip(*train_decoder_pos_data))))

            val_encoder_dense_input_data = np.array(list(map(list, zip(*val_encoder_dense_input_data))))
            val_encoder_sparse_input_data = np.array(list(map(list, zip(*val_encoder_sparse_input_data))))
            val_decoder_dense_input_data = np.array(list(map(list, zip(*val_decoder_dense_input_data))))
            val_decoder_sparse_input_data = np.array(list(map(list, zip(*val_decoder_sparse_input_data))))
            val_output_data = np.array(list(map(list, zip(*val_output_data))))
            val_weight_data = np.array(list(map(list, zip(*val_weight_data))))
            val_encoder_pos_data = np.array(list(map(list, zip(*val_encoder_pos_data))))
            val_decoder_pos_data = np.array(list(map(list, zip(*val_decoder_pos_data))))

            predict_encoder_dense_input_data = np.array(list(map(list, zip(*predict_encoder_dense_input_data))))
            predict_encoder_sparse_input_data = np.array(list(map(list, zip(*predict_encoder_sparse_input_data))))
            predict_decoder_dense_input_data = np.array(list(map(list, zip(*predict_decoder_dense_input_data))))
            predict_decoder_sparse_input_data = np.array(list(map(list, zip(*predict_decoder_sparse_input_data))))
            predict_encoder_pos_data = np.array(list(map(list, zip(*predict_encoder_pos_data))))
            predict_decoder_pos_data = np.array(list(map(list, zip(*predict_decoder_pos_data))))

            self.fit_model(_bin,
                           [train_encoder_dense_input_data, train_encoder_sparse_input_data, train_encoder_pos_data
                               , train_decoder_dense_input_data, train_decoder_sparse_input_data,
                            train_decoder_pos_data]
                           , train_output_data, train_weight_data
                           , [val_encoder_dense_input_data, val_encoder_sparse_input_data, val_encoder_pos_data
                               , val_decoder_dense_input_data, val_decoder_sparse_input_data, val_decoder_pos_data]
                           , val_output_data, val_weight_data)

            if self.args.model_predict:
                res = self.models[_bin].predict(
                    [predict_encoder_dense_input_data, predict_encoder_sparse_input_data, predict_encoder_pos_data
                        , predict_decoder_dense_input_data, predict_decoder_sparse_input_data,
                     predict_decoder_pos_data])

                item_df = self.dataset[self.dataset['date'] == self.args.cur_date][['rank', 'id', 'name']].reset_index(
                    drop=True)

                predict_date_first = (
                        datetime.datetime.strptime(self.args.cur_date, "%Y-%m-%d") + datetime.timedelta(
                    days=self.args.forecast_days_start - 1)).strftime(
                    "%Y-%m-%d")
                predict_date_last = (
                        datetime.datetime.strptime(self.args.cur_date, "%Y-%m-%d") + datetime.timedelta(
                    days=self.args.forecast_days - 1)).strftime(
                    "%Y-%m-%d")
                date_df = pd.DataFrame(
                    pd.date_range(predict_date_first, predict_date_last, freq='D').strftime("%Y-%m-%d").tolist(),
                    columns=['date'])
                item_df['key'] = 0
                date_df['key'] = 0
                res_df = pd.merge(item_df, date_df, how='left', on='key')
                res_df.sort_values(by=['rank', 'id', 'date'], ascending=[True, True, True], inplace=True)
                res_df.drop(['key', 'rank'], axis=1, inplace=True)

                res_df['forecast_value'] = res.reshape([-1])
                res_df['forecast_value'] = res_df.apply(
                    lambda x: x['forecast_value'] if x['forecast_value'] > 1 else 1, axis=1)
                res_df['forecast_value'] = (res_df['forecast_value'] - 1)

                res_df['true_value'] = pd.merge(res_df, self.dataset, how='left', on=['id','date'])['value_real']
                res_df['true_value'] = res_df.apply(lambda x: x['true_value'] if x['date']<self.data_date else -1, axis=1)

                self.res_df = res_df

    def predict_model(self, _bin, predict_x):
        return self.models[_bin][0].predict(predict_x).reshape(-1).tolist()

    def fit_model(self, _bin, train_x, train_y, train_w, val_x, val_y, val_w):
        logger.info(
            ">>>>>>>>>>{}th bin model params:".format(_bin + 1, self.model_params))
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=0.0001, verbose=1)
        earlystopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, verbose=1, mode='min',
                                      restore_best_weights=True)

        self.models[_bin].fit(list(map(lambda x: np.append(x[0], values=x[1], axis=0), zip(train_x, val_x)))
                              , np.append(train_y, values=val_y, axis=0)
                              , sample_weight=np.append(train_w, values=val_w, axis=0)
                              , epochs=100
                              , batch_size=self.model_params['batch_size']
                              , validation_data=(val_x, val_y, val_w)
                              , callbacks=[reduce_lr, earlystopping]
                              , verbose=2)

        tf.saved_model.save(self.models[_bin], 'results/models/informer_model')

    def create_model(self):
        return [self.create_informer_model() for i in range(len(self.rank_bins) - 1)]

    def create_informer_model(self):
        ##input
        encoder_dense_input = Input(shape=(self.id_nums, self.encoder_timesteps, len(self.encoder_dense_feature_cols)),
                                    name='encoder_dense_input')
        encoder_sparse_input = Input(shape=(self.id_nums, self.encoder_timesteps, len(self.encoder_sparse_feature_cols)),
                                     name='encoder_sparse_input')
        encoder_pos_input = Input(shape=(self.id_nums, self.encoder_timesteps, 1), name='encoder_pos_input')
        decoder_dense_input = Input(shape=(self.id_nums,
            self.decoder_timesteps + self.custom_params['start_token_len'], len(self.decoder_dense_feature_cols) + 1),
            name='decoder_dense_input')
        decoder_sparse_input = Input(shape=(self.id_nums,
            self.decoder_timesteps + self.custom_params['start_token_len'], len(self.decoder_sparse_feature_cols)),
            name='decoder_sparse_input')
        decoder_pos_input = Input(shape=(self.id_nums, self.decoder_timesteps + self.custom_params['start_token_len'], 1),
                                  name='decoder_pos_input')

        ##embedding
        sparse_emds = [Embedding(self.all_sparse_cols_size[i], self.model_params['embedding_size'],
                                 embeddings_regularizer=tf.keras.regularizers.l2(self.model_params['l2_reg'])) for i in
                       range(len(self.all_sparse_feature_cols))]

        pos_emds = Embedding(np.max([self.encoder_timesteps, self.custom_params['start_token_len']]) + self.decoder_timesteps, self.model_params['embedding_size'],
                             embeddings_regularizer=tf.keras.regularizers.l2(self.model_params['l2_reg']))

        encoder_sparse_emd = (Concatenate(axis=3)([sparse_emds[self.encoder_sparse_feature_index[i]](
            encoder_sparse_input[:, :, :, i]) for i in range(len(self.encoder_sparse_feature_cols))]))

        encoder_pos_emd = pos_emds(encoder_pos_input)[:,:,:,0,:]

        decoder_sparse_emd = (Concatenate(axis=3)([sparse_emds[self.decoder_sparse_feature_index[i]](
            decoder_sparse_input[:, :, :, i]) for i in range(len(self.decoder_sparse_feature_cols))]))
        decoder_pos_emd = pos_emds(decoder_pos_input)[:,:,:,0,:]
        ##encoder
        encoder_input = Dense(512, kernel_regularizer=tf.keras.regularizers.l2(self.model_params['l2_reg']))(
            Concatenate(axis=3)([encoder_dense_input, encoder_sparse_emd, encoder_pos_emd]))
        decoder_input = Dense(512, kernel_regularizer=tf.keras.regularizers.l2(self.model_params['l2_reg']))(
            Concatenate(axis=3)([decoder_dense_input, decoder_sparse_emd, decoder_pos_emd]))

        encoder_input = LayerNormalization()(encoder_input)
        encoder_input = Activation('relu')(encoder_input)
        encoder_input = Dropout(self.model_params['dropout'])(encoder_input)

        decoder_input = LayerNormalization()(decoder_input)
        decoder_input = Activation('relu')(decoder_input)
        decoder_input = Dropout(self.model_params['dropout'])(decoder_input)

        #        transformer_layer = Informer(num_layers=2, d_model=512, num_heads=8, ff_dim=512,
        #                                        output_dims=self.model_params['dense_dims'], l2_reg=self.model_params['l2_reg'])

        transformer_layer = STInformer(enc_in=self.model_params['enc_in'], dec_in=self.model_params['dec_in'], c_out=self.model_params['c_out'], seq_len=self.encoder_timesteps, label_len=0,
                                     out_len=self.decoder_timesteps, out_len_start=self.decoder_timesteps_start, batch_size=self.model_params['batch_size'],
                                     factor=self.model_params['factor'], d_model=self.model_params['d_model'], n_heads=self.model_params['n_heads'], e_layers=self.model_params['e_layers'], d_layers=self.model_params['d_layers'], d_ff=self.model_params['d_ff'],
                                     dropout=self.model_params['dropout'], attn=self.model_params['attn_type'], activation=self.model_params['activation'])

        output = transformer_layer([encoder_input, decoder_input], self.attn_mask, self.attn_mask, self.attn_mask)

        seq2seq_model = tf.keras.models.Model(
            [encoder_dense_input, encoder_sparse_input, encoder_pos_input, decoder_dense_input, decoder_sparse_input,
             decoder_pos_input],
            output)

        seq2seq_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.model_params['learning_rate']),
                              loss=self.model_params['loss'],
                              metrics=['MSE', 'MAE', 'MAPE']
                              )

#        if self.args.mode == 'debug':
#            seq2seq_model.run_eagerly = True
#        else:
#            seq2seq_model.run_eagerly = False
#
#        #        plot_model(seq2seq_model, to_file='model.png')
        return seq2seq_model


class STInformer(tf.keras.layers.Layer):

    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, out_len_start, batch_size,
                 factor=1, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512,
                 dropout=0.0, attn='prob', activation='gelu', have_enc=True):
        super(STInformer, self).__init__()
        self.pred_len = out_len
        self.pred_len_start = out_len_start
        self.attn = attn
        self.seq_len = seq_len
        self.label_len = label_len
        self.batch_size = batch_size
        self.have_enc = have_enc

        # Encoding
        # self.enc_embedding = DataEmbedding(enc_in, d_model, embed, data, dropout)
        # self.dec_embedding = DataEmbedding(dec_in, d_model, embed, data, dropout)
        # Attention
        Attn = ProbAttention if attn == 'prob' else STFullAttention
        print("attn_type:{}".format(attn))
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout),
                                   d_model, n_heads),
                    AttentionLayer(SFullAttention(False, factor, attention_dropout=dropout),
                                   d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            #           [
            #               ConvLayer(
            #                   d_model
            #               ) for l in range(e_layers - 1)
            #           ],
            norm_layer=tf.keras.layers.LayerNormalization()
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(FullAttention(True, factor, attention_dropout=dropout),
                                   d_model, n_heads),
                    AttentionLayer(SFullAttention(True, factor, attention_dropout=dropout),
                                   d_model, n_heads),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout),
                                   d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                    have_enc=have_enc
                )
                for l in range(d_layers)
            ],
            norm_layer=tf.keras.layers.LayerNormalization()
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = tf.keras.layers.Dense(c_out, name="projection_layer")

    def call(self, inputs, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        #        x_enc, x_dec, x_mark_enc, x_mark_dec = inputs
        x_enc, x_dec = inputs

        #        x_enc.set_shape((self.batch_size, self.seq_len, x_enc.shape[2]))
        #        x_mark_enc.set_shape((self.batch_size, self.seq_len, x_mark_enc.shape[2]))
        #
        #        x_dec.set_shape((self.batch_size, self.label_len+self.pred_len, x_dec.shape[2]))
        #        x_mark_dec.set_shape((self.batch_size, self.label_len+self.pred_len, x_mark_dec.shape[2]))

        #        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        #        enc_out = self.encoder(enc_out, attn_mask=enc_self_mask)
        enc_out = None
        if self.have_enc:
            enc_out = self.encoder(x_enc, attn_mask=enc_self_mask)
        #        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        #        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.decoder(x_dec, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

        dec_out = self.projection(dec_out)

        return dec_out[:, :, (-self.pred_len+self.pred_len_start-1):, :]  # [B, N, L, D]


if __name__ == '__main__':
    model = STInformer(7, 7, 7, 96, 48, 24, 32)
    x_enc = tf.zeros((32, 10, 96, 7))
    x_dec = tf.zeros((32, 10, 72, 7))
    x_mark_enc = tf.zeros((32, 10, 96, 4))
    x_mark_dec = tf.zeros((32, 10, 72, 4))
    print(model([x_enc, x_dec, x_mark_enc, x_mark_dec]).shape)

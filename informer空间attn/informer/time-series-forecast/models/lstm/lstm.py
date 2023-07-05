"""
@Time : 2023/5/9 17:34
@Author : mcxing
@File : lstm.py
@Software: PyCharm
"""

import numpy as np
import pandas as pd
import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler, KBinsDiscretizer
import json
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras.layers import Embedding, Concatenate, Flatten, Dense, Input, BatchNormalization, Activation, \
    Dropout, LSTM, GRU, Concatenate, LayerNormalization
from models.base.base_model import BaseModel
from utils.data_processing import DataGenerater
import logging
import sys
import multiprocessing

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stderr, level=logging.INFO, format=
'''[%(levelname)s] [%(asctime)s] [%(threadName)s] [%(name)s] '''
'''[%(filename)s:%(funcName)s:%(lineno)d]: %(message)s''')


class Decoder(tf.keras.layers.Layer):

    def __init__(self, latent_dim=512, dropout=0.2, dense_dims=[512, 256, 64, 8], l2_reg=0.001):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.dense_dims = dense_dims
        self.l2_reg = l2_reg
        self.lstm = LSTM(self.latent_dim, dropout=self.dropout,
                         return_sequences=True, return_state=True)
        self.denses = [Dense(num, kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg)) for num in self.dense_dims]
        self.bns = [BatchNormalization() for num in self.dense_dims]
        self.acts = [Activation('relu') for num in self.dense_dims]
        self.dropouts = [Dropout(self.dropout) for num in self.dense_dims]

        self.output_dense = Dense(1, name='output')

    def call(self, inputs):
        decoder_input = inputs[0]
        decoder_initial_state = inputs[1]
        output, decoder_state_h, decoder_state_c = self.lstm(decoder_input, initial_state=decoder_initial_state)

        ##output
        for i in range(len(self.dense_dims)):
            output = self.denses[i](output)
            output = self.bns[i](output)
            output = self.acts[i](output)
            output = self.dropouts[i](output)

        output = self.output_dense(output)

        return output, decoder_state_h, decoder_state_c


class LSTMModel(BaseModel):
    def __init__(self, args):
        super(LSTMModel, self).__init__(args)

        self.custom_params = json.loads(self.args.custom_params) if self.args.custom_params else {}
        self.train_date_gap = self.custom_params[
            'train_date_gap'] if 'train_date_gap' in self.custom_params.keys() else 1
        self.model_params['embedding_size'] = self.custom_params[
            'embedding_size'] if 'embedding_size' in self.custom_params.keys() else 32
        self.model_params['learning_rate'] = self.custom_params[
            'learning_rate'] if 'learning_rate' in self.custom_params.keys() else 0.1
        self.model_params['l2_reg'] = self.custom_params['l2_reg'] if 'l2_reg' in self.custom_params.keys() else 0.001
        self.model_params['dropout'] = self.custom_params['dropout'] if 'dropout' in self.custom_params.keys() else 0.2
        self.model_params['batch_size'] = self.custom_params[
            'batch_size'] if 'batch_size' in self.custom_params.keys() else 256
        self.model_params['latent_dim'] = self.custom_params[
            'latent_dim'] if 'latent_dim' in self.custom_params.keys() else 512
        self.model_params['dense_dims'] = self.custom_params[
            'dense_dims'] if 'dense_dims' in self.custom_params.keys() else [512, 256, 64, 8]

        self.encoder_timesteps = self.custom_params[
            'encoder_timesteps'] if 'encoder_timesteps' in self.custom_params.keys() else 30
        self.decoder_timesteps = self.args.forecast_days
        self.validation_days = self.args.validation_days

    #    def data_generater(self, id, sdf):
    #
    #        sdf.reset_index(drop=True, inplace=True)
    #        train_encoder_dense_input_data = []
    #        train_encoder_sparse_input_data = []
    #        train_decoder_dense_input_data = []
    #        train_decoder_sparse_input_data = []
    #        train_output_data = []
    #        train_weight_data = []
    #
    #        val_encoder_dense_input_data = []
    #        val_encoder_sparse_input_data = []
    #        val_decoder_dense_input_data = []
    #        val_decoder_sparse_input_data = []
    #        val_output_data = []
    #        val_weight_data = []
    #
    #        predict_encoder_dense_input_data = []
    #        predict_encoder_sparse_input_data = []
    #        predict_decoder_dense_input_data = []
    #        predict_decoder_sparse_input_data = []
    #
    #        for i in range(sdf.shape[0] - self.encoder_timesteps - 2 * self.decoder_timesteps - self.validation_days, 0,
    #                       -1):
    #            train_encoder_dense_input_data.append(
    #                [sdf.iloc[i + j][self.encoder_dense_feature_cols].values.astype(np.float32) for j in
    #                 range(self.encoder_timesteps)])
    #            train_encoder_sparse_input_data.append(
    #                [sdf.iloc[i + j][self.encoder_sparse_feature_cols].values.astype(np.float32) for j in
    #                 range(self.encoder_timesteps)])
    #            train_decoder_dense_input_data.append(
    #                [sdf.iloc[i + j + self.encoder_timesteps][self.decoder_dense_feature_cols].values.astype(np.float32) for
    #                 j in range(self.decoder_timesteps)])
    #            train_decoder_sparse_input_data.append(
    #                [sdf.iloc[i + j + self.encoder_timesteps][self.decoder_sparse_feature_cols].values.astype(np.float32)
    #                 for j in range(self.decoder_timesteps)])
    #            train_output_data.append(
    #                [sdf.iloc[i + j + self.encoder_timesteps][self.decoder_output_col].values.astype(np.float32) for j in
    #                 range(self.decoder_timesteps)])
    #            train_weight_data.append(
    #                sdf.iloc[i + self.encoder_timesteps + self.decoder_timesteps - 1]['weight'].values.astype(np.float32))
    #
    #        for i in range(sdf.shape[0] - self.encoder_timesteps - 2 * self.decoder_timesteps,
    #                       sdf.shape[0] - self.encoder_timesteps - self.decoder_timesteps - self.validation_days, -1):
    #            val_encoder_dense_input_data.append(
    #                [sdf.iloc[i + j][self.encoder_dense_feature_cols].values.astype(np.float32) for j in
    #                 range(self.encoder_timesteps)])
    #            val_encoder_sparse_input_data.append(
    #                [sdf.iloc[i + j][self.encoder_sparse_feature_cols].values.astype(np.float32) for j in
    #                 range(self.encoder_timesteps)])
    #            val_decoder_dense_input_data.append(
    #                [sdf.iloc[i + j + self.encoder_timesteps][self.decoder_dense_feature_cols].values.astype(np.float32) for
    #                 j in range(self.decoder_timesteps)])
    #            val_decoder_sparse_input_data.append(
    #                [sdf.iloc[i + j + self.encoder_timesteps][self.decoder_sparse_feature_cols].values.astype(np.float32)
    #                 for j in range(self.decoder_timesteps)])
    #            val_output_data.append(
    #                [sdf.iloc[i + j + self.encoder_timesteps][self.decoder_output_col].values.astype(np.float32) for j in
    #                 range(self.decoder_timesteps)])
    #            val_weight_data.append(
    #                sdf.iloc[i + self.encoder_timesteps + self.decoder_timesteps - 1]['weight'].values.astype(np.float32))
    #
    #        predict_encoder_dense_input_data.append([sdf.iloc[sdf.shape[0] - self.encoder_timesteps - self.decoder_timesteps + j][
    #                                                     self.encoder_dense_input_cols].values.astype(np.float32) for j in
    #                                                 range(self.encoder_timesteps)])
    #        predict_encoder_sparse_input_data.append([sdf.iloc[sdf.shape[0] - self.ncoder_timesteps - self.decoder_timesteps + j][
    #                                                      self.encoder_sparse_input_cols].values.astype(np.float32) for j in
    #                                                  range(self.encoder_timesteps)])
    #        predict_decoder_dense_input_data.append(
    #            [sdf.iloc[sdf.shape[0] - self.decoder_timesteps + j][self.decoder_dense_input_cols].values.astype(np.float32) for j in
    #             range(self.decoder_timesteps)])
    #        predict_decoder_sparse_input_data.append(
    #            [sdf.iloc[sdf.shape[0] - self.decoder_timesteps + j][self.decoder_sparse_input_cols].values.astype(np.float32) for j
    #             in range(self.decoder_timesteps)])
    #
    #        logger.info('The data of id{} are done!'.format(id))
    #        return (train_encoder_dense_input_data, train_encoder_sparse_input_data, train_decoder_dense_input_data,
    #                train_decoder_sparse_input_data, train_output_data, train_weight_data
    #                , val_encoder_dense_input_data, val_encoder_sparse_input_data, val_decoder_dense_input_data,
    #                val_decoder_sparse_input_data, val_output_data, val_weight_data
    #                , predict_encoder_dense_input_data, predict_encoder_sparse_input_data, predict_decoder_dense_input_data,
    #                predict_decoder_sparse_input_data)
    #
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
            pr = multiprocessing.Pool()
            prs = []
            for id, sdf in bin_dataset.groupby('id'):
                prs.append(pr.apply_async(DataGenerater.encoder_decoder_data_generater, args=(
                    id, sdf, self.validation_days, self.encoder_timesteps, self.decoder_timesteps,
                    self.encoder_dense_feature_cols, self.encoder_sparse_feature_cols,
                    self.decoder_dense_feature_cols, self.decoder_sparse_feature_cols,
                    self.decoder_output_col, self.train_date_gap, self.args.differential_col)))
            pr.close()
            pr.join()
            train_encoder_dense_input_data = []
            train_encoder_sparse_input_data = []
            train_decoder_dense_input_data = []
            train_decoder_sparse_input_data = []
            train_output_data = []
            train_weight_data = []
            val_encoder_dense_input_data = []
            val_encoder_sparse_input_data = []
            val_decoder_dense_input_data = []
            val_decoder_sparse_input_data = []
            val_output_data = []
            val_weight_data = []
            for x in prs:
                data = x.get()
                train_encoder_dense_input_data.extend(data[0])
                train_encoder_sparse_input_data.extend(data[1])
                train_decoder_dense_input_data.extend(data[2])
                train_decoder_sparse_input_data.extend(data[3])
                train_output_data.extend(data[4])
                train_weight_data.extend(data[5])
                val_encoder_dense_input_data.extend(data[8])
                val_encoder_sparse_input_data.extend(data[9])
                val_decoder_dense_input_data.extend(data[10])
                val_decoder_sparse_input_data.extend(data[11])
                val_output_data.extend(data[12])
                val_weight_data.extend(data[13])

            train_encoder_dense_input_data = np.array(train_encoder_dense_input_data)
            train_encoder_sparse_input_data = np.array(train_encoder_sparse_input_data)
            train_decoder_dense_input_data = np.array(train_decoder_dense_input_data)
            train_decoder_sparse_input_data = np.array(train_decoder_sparse_input_data)
            train_output_data = np.array(train_output_data)
            train_weight_data = np.array(train_weight_data)
            val_encoder_dense_input_data = np.array(val_encoder_dense_input_data)
            val_encoder_sparse_input_data = np.array(val_encoder_sparse_input_data)
            val_decoder_dense_input_data = np.array(val_decoder_dense_input_data)
            val_decoder_sparse_input_data = np.array(val_decoder_sparse_input_data)
            val_output_data = np.array(val_output_data)

            val_weight_data = np.array(val_weight_data)

            self.fit_model(_bin, [train_encoder_dense_input_data, train_encoder_sparse_input_data,
                                  train_decoder_dense_input_data, train_decoder_sparse_input_data], train_output_data,
                           train_weight_data
                           , [val_encoder_dense_input_data, val_encoder_sparse_input_data, val_decoder_dense_input_data,
                              val_decoder_sparse_input_data], val_output_data, val_weight_data)

            if self.args.model_predict:
                bin_dataset.sort_values(by=['id', 'date'], ascending=[True, True], inplace=True)
                predict_encoder_dense_input_data = []
                predict_encoder_sparse_input_data = []
                for id, sdf in bin_dataset.groupby('id'):
                    sdf.reset_index(drop=True, inplace=True)
                    predict_encoder_dense_input_data.append(
                        [sdf.iloc[sdf.shape[0] - self.encoder_timesteps - self.decoder_timesteps + j][
                             self.encoder_dense_feature_cols].values.astype(np.float32) for j in
                         range(self.encoder_timesteps)])
                    predict_encoder_sparse_input_data.append(
                        [sdf.iloc[sdf.shape[0] - self.encoder_timesteps - self.decoder_timesteps + j][
                             self.encoder_sparse_feature_cols].values.astype(np.float32) for j in
                         range(self.encoder_timesteps)])

                predict_encoder_dense_input_data = np.array(predict_encoder_dense_input_data)
                predict_encoder_sparse_input_data = np.array(predict_encoder_sparse_input_data)

                initial_states = self.models[_bin][1](
                    [predict_encoder_dense_input_data, predict_encoder_sparse_input_data])

                history_dataset = bin_dataset[bin_dataset['date'] <= self.args.cur_date]
                for i in range(self.decoder_timesteps):
                    predict_decoder_dense_input_data = []
                    predict_decoder_sparse_input_data = []
                    predict_differential_data = []
                    for id, sdf in history_dataset.groupby('id'):
                        predict_decoder_dense_input_data.append(
                            [np.append(sdf.iloc[-1][[
                                'value_advance_{}d'.format(i + 1)]].values.astype(np.float32) if 'value_advance_{}d'.format(
                                i + 1) in sdf.columns else [0],
                                       values=sdf.iloc[-1][self.decoder_dense_feature_cols].values.astype(np.float32))])
                        predict_decoder_sparse_input_data.append(
                            [sdf.iloc[-1][self.decoder_sparse_feature_cols].values.astype(np.float32)])
                        predict_differential_data.append(
                            [sdf.iloc[-(i + 2)][self.decoder_output_col].values.astype(np.float32)]
                        )

                    predict_decoder_dense_input_data = np.array(predict_decoder_dense_input_data)
                    predict_decoder_sparse_input_data = np.array(predict_decoder_sparse_input_data)
                    predict_differential_data = np.array(predict_differential_data)

                    decoder_output, h, c = self.models[_bin][2].predict(
                        [predict_decoder_dense_input_data, predict_decoder_sparse_input_data, initial_states])
                    initial_states = [h, c]

                    now_dataset = bin_dataset[bin_dataset['date'] == (
                            datetime.datetime.strptime(self.args.cur_date, "%Y-%m-%d") + datetime.timedelta(
                        days=i)).strftime("%Y-%m-%d")]
                    now_dataset[self.all_dense_feature_cols] = scaler.inverse_transform(
                        now_dataset[self.all_dense_feature_cols])
                    if self.args.differential_col is not None:
                        now_dataset['predate_value'] = np.array(
                            decoder_output.reshape(-1).tolist()) + predict_differential_data.reshape(-1)
                    else:
                        now_dataset['predate_value'] = np.array(decoder_output.reshape(-1).tolist())
                    now_dataset[self.all_dense_feature_cols] = scaler.transform(
                        now_dataset[self.all_dense_feature_cols])
                    history_dataset = history_dataset.append(now_dataset)

                history_dataset[self.all_dense_feature_cols] = scaler.inverse_transform(
                    history_dataset[self.all_dense_feature_cols])

                tdf = history_dataset[history_dataset['date'] > self.args.cur_date][
                    ['id', 'name', 'date', 'predate_value']]
                tdf.columns = ['id', 'name', 'date', 'forecast_value']
                tdf['date'] = list(
                    map(lambda x: (datetime.datetime.strptime(x, "%Y-%m-%d") + datetime.timedelta(days=-1)).strftime(
                        "%Y-%m-%d"), tdf['date']))

                tdf['forecast_value'] = list(map(lambda x: 0 if int(x) < 1 else int(x) - 1, tdf['forecast_value']))
                tdf['true_value'] = list(pd.merge(tdf, self.dataset, how='left', on=['id','date'])['value_real'])
                tdf['true_value'] = tdf.apply(lambda x: x['true_value'] if x['date']<self.data_date else -1, axis=1)



                merged_res_df = merged_res_df.append(tdf)

        self.res_df = merged_res_df

    #                ## predict results
    #                item_df = self.dataset[self.dataset['date'] == self.args.cur_date][['rank', 'id', 'name']].reset_index(
    #                    drop=True)
    #                nextdate_thelast = (
    #                        datetime.datetime.strptime(self.args.cur_date, "%Y-%m-%d") + datetime.timedelta(
    #                    days=self.args.forecast_days - 1)).strftime(
    #                    "%Y-%m-%d")
    #                date_df = pd.DataFrame(
    #                    pd.date_range(self.args.cur_date, nextdate_thelast, freq='D').strftime("%Y-%m-%d").tolist(),
    #                    columns=['date'])
    #                item_df['key'] = 0
    #                date_df['key'] = 0
    #                res_df = pd.merge(item_df, date_df, how='left', on='key')
    #                res_df.sort_values(by=['id', 'rank', 'date'], ascending=[True, True, True], inplace=True)
    #                res_df.drop(['key', 'rank'], axis=1, inplace=True)
    #
    #

    #                ## predict
    #                predict_encoder_dense_input_data = []
    #                predict_encoder_sparse_input_data = []
    #                predict_decoder_dense_input_data = []
    #                predict_decoder_sparse_input_data = []
    #
    #                for id, sdf in bin_dataset.groupby('id'):
    #                    sdf.reset_index(drop=True, inplace=True)
    #                    predict_encoder_dense_input_data.append(
    #                        [sdf.iloc[sdf.shape[0] - self.encoder_timesteps - self.decoder_timesteps + j][
    #                             self.encoder_dense_feature_cols].values.astype(np.float32) for j in
    #                         range(self.encoder_timesteps)])
    #                    predict_encoder_sparse_input_data.append(
    #                        [sdf.iloc[sdf.shape[0] - self.encoder_timesteps - self.decoder_timesteps + j][
    #                             self.encoder_sparse_feature_cols].values.astype(np.float32) for j in
    #                         range(self.encoder_timesteps)])
    #                    predict_decoder_dense_input_data.append(
    #                        [sdf.iloc[sdf.shape[0] - self.decoder_timesteps + j][
    #                            self.decoder_dense_feature_cols].values.astype(
    #                            np.float32) for j in
    #                            range(self.decoder_timesteps)])
    #                    predict_decoder_sparse_input_data.append(
    #                        [sdf.iloc[sdf.shape[0] - self.decoder_timesteps + j][
    #                            self.decoder_sparse_feature_cols].values.astype(
    #                            np.float32) for j
    #                            in range(self.decoder_timesteps)])
    #
    #                predict_encoder_dense_input_data = np.array(predict_encoder_dense_input_data)
    #                predict_encoder_sparse_input_data = np.array(predict_encoder_sparse_input_data)
    #                predict_decoder_dense_input_data = np.array(predict_decoder_dense_input_data)
    #                predict_decoder_sparse_input_data = np.array(predict_decoder_sparse_input_data)
    #
    #                res_df['value'] = list(
    #                    map(lambda x: x - 1 if x > 1 else 0, self.predict_model(_bin, [predict_encoder_dense_input_data,
    #                                                                                   predict_encoder_sparse_input_data,
    #                                                                                   predict_decoder_dense_input_data,
    #                                                                                   predict_decoder_sparse_input_data])))
    #                res_df['value'] = res_df['value'].astype('int')
    #
    #                merged_res_df = merged_res_df.append(res_df)

    def predict_model(self, _bin, predict_x):
        return self.models[_bin][0].predict(predict_x).reshape(-1).tolist()

    def fit_model(self, _bin, train_x, train_y, train_w, val_x, val_y, val_w):
        logger.info(
            ">>>>>>>>>>{}th bin model params:".format(_bin + 1, self.model_params))
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0.0001, verbose=1)
        earlystopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=8, verbose=1, mode='min',
                                      restore_best_weights=True)
        self.models[_bin][0].fit(list(map(lambda x: np.append(x[0], values=x[1], axis=0), zip(train_x, val_x)))
                                 , np.append(train_y, values=val_y, axis=0)
                                 , sample_weight=np.append(train_w, values=val_w, axis=0)
                                 , epochs=100
                                 , batch_size=self.model_params['batch_size']
                                 , validation_data=(val_x, val_y, val_w)
                                 , callbacks=[reduce_lr, earlystopping]
                                 , verbose=2)

    def create_lstm_model(self):
        ##input
        encoder_dense_input = Input(shape=(None, len(self.encoder_dense_feature_cols)), name='encoder_dense_input')
        encoder_sparse_input = Input(shape=(None, len(self.encoder_sparse_feature_cols)), name='encoder_sparse_input')
        decoder_dense_input = Input(shape=(None, len(self.decoder_dense_feature_cols) + 1), name='decoder_dense_input')
        decoder_sparse_input = Input(shape=(None, len(self.decoder_sparse_feature_cols)), name='decoder_sparse_input')

        ##embedding
        emds = [Embedding(self.all_sparse_cols_size[i], self.model_params['embedding_size'],
                          embeddings_regularizer=tf.keras.regularizers.l2(self.model_params['l2_reg'])) for i in
                range(len(self.all_sparse_feature_cols))]

        encoder_emd = (Concatenate(axis=2)([emds[self.encoder_sparse_feature_index[i]](
            encoder_sparse_input[:, :, i]) for i in range(len(self.encoder_sparse_feature_cols))]))
        decoder_emd = (Concatenate(axis=2)([emds[self.decoder_sparse_feature_index[i]](
            decoder_sparse_input[:, :, i]) for i in range(len(self.decoder_sparse_feature_cols))]))

        ##encoder
        encoder_input = Concatenate(axis=2)([encoder_dense_input, encoder_emd])
        encoder_lstm = LSTM(self.model_params['latent_dim'], dropout=self.model_params['dropout'],
                            return_sequences=True, return_state=True)
        encoder_output, state_h, state_c = encoder_lstm(encoder_input)
        encoder_states = [state_h, state_c]
        ##decoder
        decoder_input = Concatenate(axis=2)([decoder_dense_input, decoder_emd])

        #        decoder_lstm = LSTM(self.model_params['latent_dim'], dropout=self.model_params['dropout'],
        #                            return_sequences=True, return_state=True)
        #        output, decoder_state_h, decoder_state_c = decoder_lstm(decoder_input, initial_state=encoder_states)
        #
        #        ##output
        #        for num in self.model_params['dense_dims']:
        #            output = Dense(num, kernel_regularizer=tf.keras.regularizers.l2(self.model_params['l2_reg']))(output)
        #            output = BatchNormalization()(output)
        #            output = Activation('relu')(output)
        #            output = Dropout(self.model_params['dropout'])(output)
        #
        #        output = Dense(1, name='output')(output)

        decoder = Decoder(latent_dim=self.model_params['latent_dim'], dropout=self.model_params['dropout']
                          , dense_dims=self.model_params['dense_dims'], l2_reg=self.model_params['l2_reg'])

        output, _, _ = decoder([decoder_input, encoder_states])

        seq2seq_model = tf.keras.models.Model(
            [encoder_dense_input, encoder_sparse_input, decoder_dense_input, decoder_sparse_input],
            output)

        seq2seq_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.model_params['learning_rate']),
                              loss=tf.keras.losses.MeanSquaredError(),
                              metrics=['MSE', 'MAE', 'MAPE']
                              )

        #        return [seq2seq_model]

        encoder_model = tf.keras.models.Model([encoder_dense_input, encoder_sparse_input], encoder_states)
        decoder_state_input_h = Input(shape=(self.model_params['latent_dim'],))
        decoder_state_input_c = Input(shape=(self.model_params['latent_dim'],))
        decoder_states_input = [decoder_state_input_h, decoder_state_input_c]
        decoder_output, state_h, state_c = decoder([decoder_input, decoder_states_input])
        decoder_states = [state_h, state_c]

        decoder_model = tf.keras.models.Model(
            [decoder_dense_input, decoder_sparse_input] + decoder_states_input,
            [decoder_output] + decoder_states)

        #        #        encoder_model = tf.keras.model_diagrams.Model([encoder_dense_input, encoder_sparse_input],
        #                           encoder_states)
        #
        #        decoder_state_input_h = Input(shape=(self.model_params['latent_dim'],))
        #        decoder_state_input_c = Input(shape=(self.model_params['latent_dim'],))
        #        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        #        decoder_outputs, state_h, state_c = decoder_lstm(
        #            decoder_inputs, initial_state=decoder_states_inputs)
        #        decoder_states = [state_h, state_c]
        #        decoder_outputs = decoder_dense(decoder_outputs)
        #        decoder_model = Model(
        #            [decoder_inputs] + decoder_states_inputs,
        #            [decoder_outputs] + decoder_states)

        return [seq2seq_model, encoder_model, decoder_model]

    def create_model(self):
        return [self.create_lstm_model() for i in range(len(self.rank_bins) - 1)]

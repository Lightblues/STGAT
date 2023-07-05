"""
@Time : 2022/9/27 15:08
@Author : mcxing
@File : forecast_model.py
@Software: PyCharm
"""
import argparse
import numpy as np
import pandas as pd
import datetime
import math
import logging
import pyspark.sql.functions as F
from pyspark.sql.window import Window
from pyspark.sql import SparkSession
import xgboost
from xgboost.sklearn import XGBRegressor
# from lightgbm.sklearn import LGBMRegressor
# import keras_tuner as kt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, KBinsDiscretizer
import sys
import os
import json
import gc
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
import multiprocessing
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, HalvingGridSearchCV
from tensorflow.keras.layers import Embedding, Concatenate, Flatten, Dense, Input, BatchNormalization, Activation, \
    Dropout, LSTM, GRU, Concatenate
from tensorflow.keras.models import Model
from functools import reduce
import re

os.environ["PYSPARK_PYTHON"] = "/usr/bin/python3.6.5"

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stderr, level=logging.INFO, format=
'''[%(levelname)s] [%(asctime)s] [%(threadName)s] [%(name)s] '''
'''[%(filename)s:%(funcName)s:%(lineno)d]: %(message)s''')


class Utils:

    @staticmethod
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        if v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    @staticmethod
    def get_huber_loss(max_value, huber_slope_quantile):

        def huber_loss(y_true, y_pred):
            # https://en.wikipedia.org/wiki/Huber_loss#Pseudo-Huber_loss_function
            # https://www.cnblogs.com/nxf-rabbit75/p/10440805.html

            error = y_pred - y_true
#            error = np.array([x if x > 0 else x * 5 for x in error])
            scale = 1 + (error / huber_slope) ** 2
            scale_sqrt = np.sqrt(scale)
            g = error / scale_sqrt
            h = 1 / scale / scale_sqrt
            return g, h

        huber_slope = max_value * huber_slope_quantile
        huber_loss.huber_slope = huber_slope
        huber_loss.huber_slope_quantile = huber_slope_quantile

        return huber_loss

    @staticmethod
    def get_custom_mse_loss():

        def custom_mse_loss(y_true, y_pred):
            error = y_pred - y_true
            error = np.array([x if x > 0 else x * 2 for x in error])
            grad = error
            hess = np.ones_like(error)
            return grad, hess # 一阶，二阶导数
        return custom_mse_loss

    @staticmethod
    def get_huber_metric(huber_slope):
        def huber_metric(y_true, y_pred):
            loss = huber_slope ** 2 * (np.sqrt(1 + ((y_pred - y_true) / huber_slope) ** 2) - 1)
            return loss

        return huber_metric

    @staticmethod
    def huber_metric(y_pred, dtrain):
        y_true = dtrain.get_label()
        huber_slope = 100
        loss = np.mean(huber_slope ** 2 * (np.sqrt(1 + ((y_pred - y_true) / huber_slope) ** 2) - 1))
        return 'huber_loss', loss

    @staticmethod
    def spark_to_pandas(spark_df, index_col, batch_size):
        data_size = spark_df.count()
        for i in range(math.ceil(data_size / batch_size)):
            tmp_df = spark_df.where(F.col(index_col) <= (i + 1) * batch_size).where(
                F.col(index_col) > i * batch_size).toPandas()
            if i == 0:
                pandas_df = tmp_df
            else:
                pandas_df = pandas_df.append(tmp_df)
            logger.info(">>>>>>>>>>Reading data: round " + str(i + 1) + " is done! The data size is " + str(
                tmp_df.shape[0]) + "." + "The cumulative data size is " + str(pandas_df.shape[0]) + ".")
        return pandas_df

    @staticmethod
    def encoder_decoder_data_generater(id, sdf, validation_days, encoder_timesteps, decoder_timesteps,
                                       encoder_dense_feature_cols, encoder_sparse_feature_cols,
                                       decoder_dense_feature_cols, decoder_sparse_feature_cols,
                                       decoder_output_col, train_date_gap=1):

        sdf.reset_index(drop=True, inplace=True)
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

        predict_encoder_dense_input_data = []
        predict_encoder_sparse_input_data = []
        predict_decoder_dense_input_data = []
        predict_decoder_sparse_input_data = []

        for i in range(sdf.shape[0] - encoder_timesteps - 2 * decoder_timesteps - validation_days, 0,
                       -train_date_gap):
            train_encoder_dense_input_data.append(
                [sdf.iloc[i + j][encoder_dense_feature_cols].values.astype(np.float32) for j in
                 range(encoder_timesteps)])
            train_encoder_sparse_input_data.append(
                [sdf.iloc[i + j][encoder_sparse_feature_cols].values.astype(np.float32) for j in
                 range(encoder_timesteps)])
            train_decoder_dense_input_data.append(
                [sdf.iloc[i + j + encoder_timesteps][decoder_dense_feature_cols].values.astype(np.float32) for
                 j in range(decoder_timesteps)])
            train_decoder_sparse_input_data.append(
                [sdf.iloc[i + j + encoder_timesteps][decoder_sparse_feature_cols].values.astype(np.float32)
                 for j in range(decoder_timesteps)])
            train_output_data.append(
                [sdf.iloc[i + j + encoder_timesteps][decoder_output_col].values.astype(np.float32) for j in
                 range(decoder_timesteps)])
            train_weight_data.append(
                sdf.iloc[i + encoder_timesteps + decoder_timesteps - 1]['weight'].astype(np.float32))

        for i in range(sdf.shape[0] - encoder_timesteps - 2 * decoder_timesteps,
                       sdf.shape[0] - encoder_timesteps - 2 * decoder_timesteps - validation_days, -1):
            val_encoder_dense_input_data.append(
                [sdf.iloc[i + j][encoder_dense_feature_cols].values.astype(np.float32) for j in
                 range(encoder_timesteps)])
            val_encoder_sparse_input_data.append(
                [sdf.iloc[i + j][encoder_sparse_feature_cols].values.astype(np.float32) for j in
                 range(encoder_timesteps)])
            val_decoder_dense_input_data.append(
                [sdf.iloc[i + j + encoder_timesteps][decoder_dense_feature_cols].values.astype(np.float32) for
                 j in range(decoder_timesteps)])
            val_decoder_sparse_input_data.append(
                [sdf.iloc[i + j + encoder_timesteps][decoder_sparse_feature_cols].values.astype(np.float32)
                 for j in range(decoder_timesteps)])
            val_output_data.append(
                [sdf.iloc[i + j + encoder_timesteps][decoder_output_col].values.astype(np.float32) for j in
                 range(decoder_timesteps)])
            val_weight_data.append(
                sdf.iloc[i + encoder_timesteps + decoder_timesteps - 1]['weight'].astype(np.float32))

#       predict_encoder_dense_input_data.append([sdf.iloc[sdf.shape[0] - encoder_timesteps - decoder_timesteps + j][
#                                                    encoder_dense_feature_cols].values.astype(np.float32) for j in
#                                                range(encoder_timesteps)])
#       predict_encoder_sparse_input_data.append([sdf.iloc[sdf.shape[0] - encoder_timesteps - decoder_timesteps + j][
#                                                     encoder_sparse_feature_cols].values.astype(np.float32) for j in
#                                                 range(encoder_timesteps)])
#       predict_decoder_dense_input_data.append(
#           [sdf.iloc[sdf.shape[0] - decoder_timesteps + j][decoder_dense_feature_cols].values.astype(np.float32) for j in
#            range(decoder_timesteps)])
#       predict_decoder_sparse_input_data.append(
#           [sdf.iloc[sdf.shape[0] - decoder_timesteps + j][decoder_sparse_feature_cols].values.astype(np.float32) for j
#            in range(decoder_timesteps)])

        logger.info('The data of id{} are done!'.format(id))
        return (train_encoder_dense_input_data, train_encoder_sparse_input_data, train_decoder_dense_input_data,
                train_decoder_sparse_input_data, train_output_data, train_weight_data
                , val_encoder_dense_input_data, val_encoder_sparse_input_data, val_decoder_dense_input_data,
                val_decoder_sparse_input_data, val_output_data, val_weight_data
#               , predict_encoder_dense_input_data, predict_encoder_sparse_input_data, predict_decoder_dense_input_data,
#               predict_decoder_sparse_input_data
                )

    @staticmethod
    def get_args(mode="run", debug_args={}):
        logger.info(">>>>>>>>>>Start reading configurations:")
        parser = argparse.ArgumentParser()
        parser.add_argument("--model_class", type=str, required=True)
        parser.add_argument("--model_name", type=str, required=True)
        parser.add_argument("--model_type", type=str, required=False, default='multi_model')
        parser.add_argument("--output_type", type=str, required=False, default='single_output')
        parser.add_argument("--cur_date", type=str, required=True)
        parser.add_argument("--dataset_table", type=str, required=True)
        parser.add_argument("--export_table", type=str, required=True)
        parser.add_argument("--params_table", type=str, required=True)
        parser.add_argument("--sparse_feature_cols", type=str, nargs='+', required=False, default=[])
        parser.add_argument("--normal_feature_cols", type=str, nargs='+', required=False, default=[])
        parser.add_argument("--shift_feature_cols", type=str, nargs='+', required=False, default=[])

        parser.add_argument("--encoder_sparse_feature_cols", type=str, nargs='+', required=False, default=[])
        parser.add_argument("--encoder_dense_feature_cols", type=str, nargs='+', required=False, default=[])
        parser.add_argument("--decoder_sparse_feature_cols", type=str, nargs='+', required=False, default=[])
        parser.add_argument("--decoder_dense_feature_cols", type=str, nargs='+', required=False, default=[])

        parser.add_argument("--target_col", type=str, required=True)
        parser.add_argument("--weight_half_life", type=int, required=False, default=30)
        parser.add_argument("--weight_minimum", type=float, required=False, default=1)
        parser.add_argument("--multioutput_type", type=str, required=False, default='RegressorChain')
        parser.add_argument("--forecast_days", type=int, required=False, default=30)
        parser.add_argument("--validation_days", type=int, required=False, default=0)
        parser.add_argument("--huber_slope_quantile", type=float, required=False, default=1)
        parser.add_argument("--date_col", type=str, required=False, default='date')
        parser.add_argument("--id_col", type=str, required=False, default='id')
        parser.add_argument("--name_col", type=str, required=False, default='name')
        parser.add_argument("--rank_col", type=str, required=False, default='rk')
        parser.add_argument("--rank_limit", type=int, required=False, default=0)
        parser.add_argument("--rank_bin_length", type=int, required=False, default=0)
        parser.add_argument("--rank_bins", type=int, nargs='+', required=False, default=[])
        parser.add_argument("--custom_params", type=str, required=False, default={})
        parser.add_argument("--model_load", type=Utils.str2bool, required=False, default=False)
        parser.add_argument("--model_tune", type=Utils.str2bool, required=False, default=False)
        parser.add_argument("--model_predict", type=Utils.str2bool, required=False, default=True)
        parser.add_argument("--params_to_tune", type=str, nargs='+', required=False, default=[])
        parser.add_argument("--candidate_params", type=str, required=False, default={})

        if mode == 'debug':
            return parser.parse_args(debug_args)

        return parser.parse_args()


class BaseModel:
    def __init__(self, args):
        self.args = args
        logger.info(">>>>>>>>>>args:{}:".format(self.args))
        self.spark = self.get_spark()
        self.max_value = float('inf')
        self.dataset = None
        self.loaded_params = None
        self.rank_bins = None
        self.models = None
        self.res_df = None
        self.best_params = []
        if self.args.custom_params:
            self.custom_params = json.loads(self.args.custom_params)
        else:
            self.custom_params = {}
        logger.info(">>>>>>>>>>Custom model params: {}".format(self.custom_params))

        self.x_num = None
        self.model_params = {}

    def get_spark(self):
        spark = SparkSession.builder.appName("forecast_model_{}".format(self.args.model_name)).config(
            'spark.sql.autoBroadcastJoinThreshold', '-1').config(
            'spark.shuffle.service.enabled', 'true').config(
            'spark.kryoserializer.buffer.max', '2000m').config(
            'spark.dynamicAllocation.enabled', 'true').config(
            'spark.executor.memory', '8g').config(
            'spark.driver.memory', '8g').config(
            'spark.driver.maxResultSize', '0').enableHiveSupport().getOrCreate()
        return spark

    def read_data(self):
        logger.info(">>>>>>>>>>Start reading data:")

        max_d = self.spark.table(self.args.dataset_table).select(F.col("d")).orderBy(F.col("d").desc()).limit(
            1).toPandas().iloc[0][0]
        dataset_sdf = self.spark.table(self.args.dataset_table).where(F.col('d') == self.args.cur_date)
        all_cols = list(
            set(self.args.sparse_feature_cols + self.args.normal_feature_cols + self.args.shift_feature_cols + [
                self.args.target_col] + [self.args.date_col] + [self.args.id_col] + [self.args.name_col] + [
                    self.args.rank_col]).intersection(set(dataset_sdf.columns)))
        logger.info(">>>>>>>>>>All columns: {}".format(all_cols))
        dataset_sdf = dataset_sdf.select(all_cols)

        if self.args.rank_bins:
            dataset_sdf = dataset_sdf.where(F.col(self.args.rank_col) <= self.args.rank_bins[-1]).where(
                F.col(self.args.rank_col) >= self.args.rank_bins[0])
        elif self.args.rank_limit:
            dataset_sdf = dataset_sdf.where(F.col(self.args.rank_col) <= self.args.rank_limit)

        dataset_sdf = dataset_sdf.withColumn("index", F.row_number().over(Window().orderBy(F.lit("A")))).persist()
        dataset = Utils.spark_to_pandas(dataset_sdf, "index", 1000000)
        ##列名标准化
        dataset.rename(columns={self.args.date_col: "date", self.args.id_col: "id", self.args.name_col: "name",
                                self.args.rank_col: "rank", self.args.target_col: "value"}, inplace=True)
        #        dataset.rename(columns=lambda x: x.replace(self.args.value_col, 'value'), inplace=True)
        if 'rank' not in dataset.columns:
            logger.info(">>>>>>>>>>There is no rank colum.")
            dataset['rank'] = 1

        dataset['weight'] = dataset['date'].apply(
            lambda x: max(self.args.weight_minimum
                          ,
                          math.pow(math.e,
                                   -0.693147 / self.args.weight_half_life * (
                                           datetime.datetime.strptime(
                                               self.args.cur_date,
                                               "%Y-%m-%d") - datetime.datetime.strptime(
                                       x,
                                       "%Y-%m-%d")).days)
                          ))

        dataset.sort_values(by=['date', 'rank', 'id'], ascending=[True, True, True], inplace=True)
        self.max_value = dataset['value'].max()
        self.dataset = dataset
        self.rank_bins = self.get_rank_bins()
        logger.info(">>>>>>>>>>rank_bins:{}:".format(self.rank_bins))

        if self.args.model_load:
            params_d = self.spark.table(self.args.params_table).where(F.col('d') < self.args.cur_date).select(
                F.col("d")).orderBy(
                F.col("d").desc()).limit(1).toPandas()

            if params_d.shape[0] > 0:
                max_d = params_d.iloc[0][0]
                self.loaded_params = self.spark.table(self.args.params_table).where(F.col('d') == max_d).where(
                    F.col('model') == re.search(r'^(.*?)(_test.*?)?$', self.args.model_name, re.M | re.I).group(
                        1)).toPandas()
            else:
                logger.info(">>>>>>>>>>There is no params to load!")
                self.args.model_load = False

    def get_rank_bins(self):
        if self.args.rank_bins:
            return self.args.rank_bins
        if self.args.rank_limit == 0:
            min_rank_limit = self.dataset['rank'].max()
        else:
            min_rank_limit = min(self.args.rank_limit, self.dataset['rank'].max())

        if self.args.rank_bin_length == 0:
            rank_bins = [0, min_rank_limit]
        else:
            rank_bins = list(range(0, min_rank_limit, self.args.rank_bin_length))
            rank_bins.append(min_rank_limit)

        return rank_bins

    def create_model(self):
        pass

    def tune_model(self):
        pass

    def fit_model(self):
        pass

    def predict_model(self):
        pass

    def train_model_multi_output(self):
        logger.info(">>>>>>>>>>Start training model:")
        if self.dataset is None:
            raise Exception(">>>>>>>>>>There is no data!")
        self.models = self.create_model()
        merged_res_df = pd.DataFrame()
        for _bin in range(len(self.rank_bins) - 1):
            bin_dataset = self.dataset[
                (self.dataset['rank'] > self.rank_bins[_bin]) & (self.dataset['rank'] <= self.rank_bins[_bin + 1])]
            logger.info(
                ">>>>>>>>>>{}th bin {} starts".format(_bin + 1, [self.rank_bins[_bin], self.rank_bins[_bin + 1]]))
            now_dataset = bin_dataset[
                self.args.normal_feature_cols + ['id', 'date', 'weight']]
            for _day in range(self.args.forecast_days):
                shift_df = bin_dataset.groupby(['id'])[self.args.shift_feature_cols + ['value']].shift(-_day)
                shift_df.columns = shift_df.columns.map(lambda x: x + '_' + str(_day))
                now_dataset = pd.concat([now_dataset, shift_df], axis=1)

            if 'covid19' in now_dataset.columns:
                now_dataset = now_dataset[now_dataset['covid19'] == 0].dropna()
            else:
                now_dataset = now_dataset.dropna()
            ## train
            val_split_date = (datetime.datetime.strptime(model.args.cur_date, "%Y-%m-%d") + datetime.timedelta(
                days=-(model.args.validation_days + model.args.forecast_days))).strftime("%Y-%m-%d")
            val_end_date = (datetime.datetime.strptime(model.args.cur_date, "%Y-%m-%d") + datetime.timedelta(
                days=-(model.args.forecast_days))).strftime("%Y-%m-%d")
            logger.info(">>>>>>>>>>val_split_date:{}".format(val_split_date))
            logger.info(">>>>>>>>>>val_end_date:{}".format(val_end_date))

            now_train_dataset = now_dataset[now_dataset['date'] < val_split_date]
            now_train_y = now_train_dataset[['value_' + str(_day) for _day in range(model.args.forecast_days)]].values
            now_train_x = now_train_dataset.drop(
                ['id', 'date', 'weight'] + ['value_' + str(_day) for _day in range(model.args.forecast_days)],
                axis=1).values
            now_train_w = now_train_dataset['weight'].values

            now_val_dataset = now_dataset[
                (now_dataset['date'] >= val_split_date) & (now_dataset['date'] <= val_end_date)]
            now_val_y = now_val_dataset[['value_' + str(_day) for _day in range(model.args.forecast_days)]].values
            now_val_x = now_val_dataset.drop(
                ['id', 'date', 'weight'] + ['value_' + str(_day) for _day in range(model.args.forecast_days)],
                axis=1).values
            now_val_w = now_val_dataset['weight'].values

            if self.args.model_tune:
                self.tune_model(_bin, 0, now_train_x, now_train_y, now_train_w)

            self.fit_model(_bin, 0, now_train_x, now_train_y, now_train_w, now_val_x, now_val_y, now_val_w)

            if self.args.model_predict:
                ## predict results
                item_df = self.dataset[self.dataset['date'] == self.args.cur_date][['rank', 'id', 'name']].reset_index(
                    drop=True)
                nextdate_thelast = (
                        datetime.datetime.strptime(self.args.cur_date, "%Y-%m-%d") + datetime.timedelta(
                    days=self.args.forecast_days - 1)).strftime(
                    "%Y-%m-%d")
                date_df = pd.DataFrame(
                    pd.date_range(self.args.cur_date, nextdate_thelast, freq='D').strftime("%Y-%m-%d").tolist(),
                    columns=['date'])
                item_df['key'] = 0
                date_df['key'] = 0
                res_df = pd.merge(item_df, date_df, how='left', on='key')
                res_df.sort_values(by=['rank', 'id', 'date'], ascending=[True, True, True], inplace=True)
                res_df.drop(['key', 'rank'], axis=1, inplace=True)

                ## predict
                now_predict_dataset = now_dataset[now_dataset['date'] == self.args.cur_date]
                now_predict_x = now_predict_dataset.drop(
                    ['id', 'date', 'weight'] + ['value_' + str(_day) for _day in range(self.args.forecast_days)],
                    axis=1).values

                res_df['value'] = list(
                    map(lambda x: x - 1 if x > 1 else 0, self.predict_model(_bin, 0, now_predict_x)))
                res_df['value'] = res_df['value'].astype('int')

                merged_res_df = merged_res_df.append(res_df)

                self.res_df = merged_res_df

    def train_model(self):
        logger.info(">>>>>>>>>>Start training model:")
        logger.info(">>>>>>>>>>model_type:{}, output_type:{}".format(self.args.model_type, self.args.output_type))

        if self.dataset is None:
            raise Exception(">>>>>>>>>>There is no data!")
        self.models = self.create_model()
        merged_dataset = pd.DataFrame()
        for _bin in range(len(self.rank_bins) - 1):
            bin_dataset = self.dataset[
                (self.dataset['rank'] > self.rank_bins[_bin]) & (self.dataset['rank'] <= self.rank_bins[_bin + 1])]
            logger.info(
                ">>>>>>>>>>{}th bin {} starts".format(_bin + 1, [self.rank_bins[_bin], self.rank_bins[_bin + 1]]))
            for _day in range(self.args.forecast_days):
                logger.info(">>>>>>>>>>{}th bin {}th model starts:".format(_bin + 1, _day + 1))
                ## ith dataset
                predict_value_col = 'nextdate' + str(_day) + '_predict_value'
                predict_feature_cols = []
                if self.args.multioutput_type == 'RegressorChain':
                    predict_feature_cols = ['nextdate' + str(j) + '_predict_value' for j in range(_day)]
                normal_feature_df = bin_dataset[
                    self.args.normal_feature_cols + predict_feature_cols + ['id', 'date', 'weight']]

                shift_df = bin_dataset.groupby(['id'])[self.args.shift_feature_cols + ['value']].shift(
                    -_day)

                now_dataset = pd.concat([normal_feature_df, shift_df], axis=1)

                now_dataset = now_dataset[now_dataset['covid19'] == 0].dropna()
                if 'holiday_type' in bin_dataset.columns:
                    now_dataset['holiday_type'] = now_dataset['holiday_type'].astype(int)
                now_dataset = pd.get_dummies(now_dataset, columns=self.args.sparse_feature_cols)

                ## train
                val_split_date = (datetime.datetime.strptime(self.args.cur_date, "%Y-%m-%d") + datetime.timedelta(
                    days=-(self.args.validation_days + _day))).strftime("%Y-%m-%d")
                val_end_date = (datetime.datetime.strptime(self.args.cur_date, "%Y-%m-%d") + datetime.timedelta(
                    days=-_day)).strftime("%Y-%m-%d")
                logger.info(">>>>>>>>>>val_split_date:{}".format(val_split_date))
                logger.info(">>>>>>>>>>val_end_date:{}".format(val_end_date))

                now_train_dataset = now_dataset[now_dataset['date'] < val_split_date]
                now_train_y = now_train_dataset['value'].values
                now_train_x = now_train_dataset.drop(['id', 'date', 'weight', 'value'], axis=1).values
                now_train_w = now_train_dataset['weight'].values

                now_val_dataset = now_dataset[
                    (now_dataset['date'] >= val_split_date) & (now_dataset['date'] < val_end_date)]
                now_val_y = now_val_dataset['value'].values
                now_val_x = now_val_dataset.drop(['id', 'date', 'weight', 'value'], axis=1).values
                now_val_w = now_val_dataset['weight'].values

                self.x_num = now_train_x.shape[1]

                if self.args.model_tune:
                    self.tune_model(_bin, _day, now_train_x, now_train_y, now_train_w, now_val_x, now_val_y, now_val_w)

                self.fit_model(_bin, _day, now_train_x, now_train_y, now_train_w, now_val_x, now_val_y, now_val_w)
                ## predict
                now_predict_dataset = now_dataset[now_dataset['date'] <= self.args.cur_date]
                now_predict_x = now_predict_dataset.drop(['id', 'date', 'weight', 'value'], axis=1).values
                now_predict_dataset[predict_value_col] = list(
                    map(lambda x: x if x > 1 else 1, list(self.predict_model(_bin, _day, now_predict_x))))

                bin_dataset = pd.merge(bin_dataset, now_predict_dataset[['id', 'date', predict_value_col]],
                                       how='left',
                                       on=['id', 'date'])

            merged_dataset = merged_dataset.append(bin_dataset)
            #            del bin_dataset
            gc.collect()

        if self.args.model_predict:
            ## predict results
            self.dataset = merged_dataset
            item_df = self.dataset[self.dataset['date'] == self.args.cur_date][['rank', 'id', 'name']].reset_index(
                drop=True)
            nextdate_thelast = (
                    datetime.datetime.strptime(self.args.cur_date, "%Y-%m-%d") + datetime.timedelta(
                days=self.args.forecast_days - 1)).strftime(
                "%Y-%m-%d")
            date_df = pd.DataFrame(
                pd.date_range(self.args.cur_date, nextdate_thelast, freq='D').strftime("%Y-%m-%d").tolist(),
                columns=['date'])
            item_df['key'] = 0
            date_df['key'] = 0
            res_df = pd.merge(item_df, date_df, how='left', on='key')
            res_df.sort_values(by=['rank', 'id', 'date'], ascending=[True, True, True], inplace=True)
            res_df.drop(['key', 'rank'], axis=1, inplace=True)

            predict_feature_cols = []
            for i in range(self.args.forecast_days):
                predict_feature_cols += ['nextdate' + str(i) + '_predict_value']
            res_df['value'] = self.dataset[self.dataset['date'] == self.args.cur_date][
                predict_feature_cols].values.reshape(
                [-1, 1])
            res_df['value'] = (res_df['value'] - 1).astype('int')

            self.res_df = res_df

    def save_data(self):
        logger.info(">>>>>>>>>>Start saving data:")
        if self.res_df is None:
            raise Exception(">>>>>>>>>>There is no predict results!")

        spark_df = self.spark.createDataFrame(self.res_df)
        spark_df.createOrReplaceTempView('table_temp')
        pre_date = (datetime.datetime.strptime(self.args.cur_date, "%Y-%m-%d") + datetime.timedelta(days=-1)).strftime(
            "%Y-%m-%d")
        if self.args.export_table.split('.')[0][:3] == 'tmp':
            self.spark.sql("DROP TABLE IF EXISTS {0} ".format(self.args.export_table))
            self.spark.sql("CREATE TABLE {0} SELECT * FROM table_temp".format(self.args.export_table))
        else:
            self.spark.sql(
                "INSERT OVERWRITE table {0} PARTITION (d='{1}', model='{2}')SELECT * FROM table_temp".format(
                    self.args.export_table, pre_date, self.args.model_name))

    def save_params(self):
        logger.info(">>>>>>>>>>Start saving model params:")
        best_params_df = pd.DataFrame(self.best_params, columns=['bin_number', 'model_number', 'best_params'])
        spark_df = self.spark.createDataFrame(best_params_df)
        spark_df.createOrReplaceTempView('table_temp')
        pre_date = (datetime.datetime.strptime(self.args.cur_date, "%Y-%m-%d") + datetime.timedelta(days=-1)).strftime(
            "%Y-%m-%d")
        if self.args.export_table.split('.')[0][:3] == 'tmp':
            self.spark.sql("DROP TABLE IF EXISTS {0} ".format(self.args.params_table))
            self.spark.sql("CREATE TABLE {0} SELECT * FROM table_temp".format(self.args.params_table))
        else:
            self.spark.sql(
                "INSERT OVERWRITE table {0} PARTITION (d='{1}', model='{2}')SELECT * FROM table_temp".format(
                    self.args.params_table, pre_date, self.args.model_name))

    def pipline(self):
        logger.info(">>>>>>>>>>Pipline starts.")
        self.read_data()

        if self.args.output_type == 'multi_output':
            self.train_model_multi_output()
        else:
            self.train_model()

        if self.args.model_tune:
            self.save_params()

        if self.args.model_predict:
            self.save_data()

        logger.info(">>>>>>>>>>Pipline ends.")


class XgbModel(BaseModel):
    def __init__(self, args):
        super(XgbModel, self).__init__(args)

    def create_model(self):
        models = [[XGBRegressor(n_estimators=50,
                                max_depth=11,
                                learning_rate=0.1,
                                verbosity=3,
#                                objective=Utils.get_huber_loss(self.max_value, self.args.huber_slope_quantile),
                                objective=Utils.get_custom_mse_loss(),
                                booster='gbtree',
                                tree_method='exact',
                                gamma=0,
                                min_child_weight=6,
                                max_delta_step=0,
                                subsample=0.8,
                                colsample_bytree=0.8,
                                colsample_bylevel=1,
                                colsample_bynode=1,
                                reg_alpha=1,
                                reg_lambda=1,
                                scale_pos_weight=1,
                                random_state=0,
                                missing=-99999
                                ) for i in range(self.args.forecast_days)] for j in range(len(self.rank_bins) - 1)]



        logger.info(">>>>>>>>>>Default model params: {}".format(models[0][0].get_params))

        if self.args.model_load:
            for _bin in range(len(self.rank_bins) - 1):
                for _day in range(self.args.forecast_days):
                    best_params_list = self.loaded_params[
                        (self.loaded_params['rank_bin_number'].astype(int) == _bin + 1) & (
                                self.loaded_params['model_number'].astype(int) == _day + 1)]['best_params'].values
                    if best_params_list.size > 0:
                        best_params = json.loads(best_params_list[0])
                        if 'huber_slope_quantile' in best_params.keys():
                            best_params['objective'] = Utils.get_huber_loss(self.max_value,
                                                                            best_params['huber_slope_quantile'])
                        logger.info(
                            ">>>>>>>>>>{}th bin {}th model loaded params: {}".format(_bin + 1, _day + 1, best_params))
                        models[_bin][_day].set_params(**best_params)
                    else:
                        logger.info(
                            ">>>>>>>>>>There is no {}th bin {}th model loaded params!".format(_bin + 1, _day + 1))
                    if self.custom_params:
                        models[_bin][_day].set_params(**self.custom_params)
        return models

    def tune_model(self, _bin, _day, train_x, train_y, train_w, val_x, val_y, val_w):

        def tune_params(model, params, cv, method='HalvingGridSearchCV'):
            if params:
                if method == 'GridSearchCV':
                    clf = GridSearchCV(model, params, cv=cv, verbose=0, n_jobs=8, scoring='neg_mean_squared_error',
                                       return_train_score=False, min_resources='exhaust', refit=False)
                else:
                    clf = HalvingGridSearchCV(model, params, cv=cv, verbose=0, n_jobs=8,
                                              scoring='neg_mean_squared_error',
                                              return_train_score=False, factor=3, refit=False, random_state=2022)
                clf.fit(train_x, train_y, sample_weight=train_w)
                model.set_params(**clf.best_params_)
                logging.info('best_params:{}'.format(clf.best_params_))
                if 'objective' in clf.best_params_ and clf.best_params_['objective'].__name__ == 'huber_loss':
                    model.set_params(**{'huber_slope_quantile': clf.best_params_['objective'].huber_slope_quantile})
                    logging.info(
                        'best_params-huber_slope_quantile:{}'.format(
                            clf.best_params_['objective'].huber_slope_quantile))
                gc.collect()

            return model

        logger.info(">>>>>>>>>>Start {}th bin {}th model tune:".format(_bin + 1, _day + 1))
        cv_n_splits = self.custom_params['cv_n_splits'] if 'cv_n_splits' in self.custom_params.keys() else 5
        logger.info(">>>>>>>>>>cv_n_splits:{}".format(cv_n_splits))
        tscv = TimeSeriesSplit(n_splits=cv_n_splits, test_size=len(self.dataset['id'].unique()))
        o1 = Utils.get_huber_loss(model.max_value, 0.8)
        o2 = Utils.get_huber_loss(model.max_value, 0.9)
        o3 = Utils.get_huber_loss(model.max_value, 0.95)
        o4 = Utils.get_huber_loss(model.max_value, 0.99)
        o5 = Utils.get_huber_loss(model.max_value, 1)

        candidate_params = {'n_estimators': [10, 20, 50, 100, 150, 200, 300, 400],
                            'learning_rate': [0.01, 0.02, 0.05, 0.1],
                            'max_depth': [5, 7, 9, 11, 13],
                            'min_child_weight': [3, 5, 7, 9, 11],
                            'gamma': [0, 0.01, 0.05, 0.1, 1],
                            'subsample': [0.7, 0.8, 0.85, 0.9, 1],
                            'colsample_bytree': [0.7, 0.8, 0.85, 0.9, 1],
                            'reg_alpha': [0, 0.5, 1, 1.5, 2],
                            'reg_lambda': [0, 0.5, 1, 1.5, 2],
                            'objective': [o1, o2, o3, o4, o5],
                            'tree_method': ['exact', 'approx', 'hist', 'auto'],
                            'max_bin': [64, 128, 256, 512, 1024, 2048]
                            }
        if self.args.candidate_params:
            candidate_params.update(json.loads(self.args.candidate_params))
        self.models[_bin][_day].set_params(**{'n_jobs': multiprocessing.cpu_count() // 8, 'verbosity': 0})

        ##max_depth, min_child_weight
        self.models[_bin][_day] = tune_params(model=self.models[_bin][_day], params=dict(
            (key, value) for key, value in candidate_params.items() if
            key in ('max_depth', 'min_child_weight') and key in self.args.params_to_tune), cv=tscv)
        ##gamma
        self.models[_bin][_day] = tune_params(model=self.models[_bin][_day], params=dict(
            (key, value) for key, value in candidate_params.items() if
            key in ('gamma') and key in self.args.params_to_tune), cv=tscv)
        ##subsample, colsample_bytree
        self.models[_bin][_day] = tune_params(model=self.models[_bin][_day], params=dict(
            (key, value) for key, value in candidate_params.items() if
            key in ('subsample', 'colsample_bytree') and key in self.args.params_to_tune), cv=tscv)
        ##reg_alpha, reg_lambda
        self.models[_bin][_day] = tune_params(model=self.models[_bin][_day], params=dict(
            (key, value) for key, value in candidate_params.items() if
            key in ('reg_alpha', 'reg_lambda') and key in self.args.params_to_tune), cv=tscv)
        ##huber_slope
        self.models[_bin][_day] = tune_params(model=self.models[_bin][_day], params=dict(
            (key, value) for key, value in candidate_params.items() if
            key in ('objective') and 'huber_slope' in self.args.params_to_tune), cv=tscv)
        ##tree_method
        self.models[_bin][_day] = tune_params(model=self.models[_bin][_day], params=dict(
            (key, value) for key, value in candidate_params.items() if
            key in ('tree_method') and key in self.args.params_to_tune), cv=tscv)
        ##max_bin
        self.models[_bin][_day] = tune_params(model=self.models[_bin][_day], params=dict(
            (key, value) for key, value in candidate_params.items() if
            key in ('max_bin') and key in self.args.params_to_tune and self.models[_bin][_day].get_params()[
                'tree_method'] in ('hist', 'gpu_hist')), cv=tscv)
        ##n_estimators, learning_rate
        #        self.models[_bin][_day] = tune_params(model=self.models[_bin][_day], params=dict(
        #            (key, value) for key, value in candidate_params.items() if
        #            key in ('n_estimators', 'learning_rate') and key in self.args.params_to_tune), cv=tscv)

        self.models[_bin][_day] = tune_params(model=self.models[_bin][_day], params=dict(
            (key, value) for key, value in candidate_params.items() if
            key in ('n_estimators') and key in self.args.params_to_tune), cv=tscv)
        self.models[_bin][_day] = tune_params(model=self.models[_bin][_day], params=dict(
            (key, value) for key, value in candidate_params.items() if
            key in ('learning_rate') and key in self.args.params_to_tune), cv=tscv)

        self.models[_bin][_day].set_params(**{'n_jobs': multiprocessing.cpu_count(), 'verbosity': 3})
        params = self.models[_bin][_day].get_params()
        logger.info(">>>>>>>>>>{}th bin {} model best params: {}".format(_bin + 1, _day + 1, params))
        if params['objective'].__name__ == 'huber_loss':
            del params['objective']
        self.best_params.append([_bin + 1, _day + 1, json.dumps(params)])

    def fit_model(self, _bin, _day, train_x, train_y, train_w, val_x, val_y, val_w):
        logger.info(">>>>>>>>>>Start {}th bin {}th model fit:".format(_bin + 1, _day + 1))
        if self.args.validation_days > 0:
            es = xgboost.callback.EarlyStopping(
                rounds=10,
                min_delta=1e-3,
                save_best=True,
                maximize=False,
                data_name="validation_0",
                # metric_name="rmse",
            )
            self.models[_bin][_day].fit(train_x, train_y, sample_weight=train_w
                                        , eval_set=[(val_x, val_y)], sample_weight_eval_set=[(val_w)]
                                       , eval_metric='rmse'
                                        , callbacks=[es]
                                        , verbose=True)
            logger.info(">>>>>>>>>>{}th bin {} model best iteration: {}".format(_bin + 1, _day + 1,
                                                                                self.models[_bin][_day].best_iteration))
            self.models[_bin][_day].set_params(**{'n_estimators': self.models[_bin][_day].best_iteration})
            self.models[_bin][_day].fit(np.append(train_x, values=val_x, axis=0),
                                        np.append(train_y, values=val_y, axis=0),
                                        sample_weight=np.append(train_w, values=val_w, axis=0), verbose=True)
        else:
            self.models[_bin][_day].fit(train_x, train_y, sample_weight=train_w, verbose=True)

    def predict_model(self, _bin, _day, predict_x):
        logger.info(">>>>>>>>>>Start {}th bin {}th model predict:".format(_bin + 1, _day + 1))
        return self.models[_bin][_day].predict(predict_x)


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
        #            self.models[_bin][_day].fit(val_x, [val_x, val_y, val_y]
        #                                        , epochs=len(self.history[_bin][_day].epoch) - 6
        #                                        , batch_size=256
        #                                        , verbose=2
        #                                        , sample_weight=[val_w, val_w, val_w]
        #                                        )
        else:
            #            self.history[_bin][_day] = self.models[_bin][_day].fit(train_x
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

        self.encoder_dense_feature_cols = self.args.encoder_dense_feature_cols
        self.encoder_sparse_feature_cols = self.args.encoder_sparse_feature_cols
        self.decoder_dense_feature_cols = self.args.decoder_dense_feature_cols
        self.decoder_sparse_feature_cols = self.args.decoder_sparse_feature_cols
        self.decoder_output_col = ['value_real']

        self.all_dense_feature_cols = list(set(self.encoder_dense_feature_cols + self.decoder_dense_feature_cols))
        self.all_sparse_feature_cols = list(set(self.encoder_sparse_feature_cols + self.decoder_sparse_feature_cols))

        self.encoder_sparse_feature_index = list(map(lambda x: self.all_sparse_feature_cols.index(x), self.encoder_sparse_feature_cols))
        self.decoder_sparse_feature_index = list(map(lambda x: self.all_sparse_feature_cols.index(x), self.decoder_sparse_feature_cols))



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
        end_date = (datetime.datetime.strptime(self.args.cur_date, "%Y-%m-%d") + datetime.timedelta(
            days=(self.args.forecast_days-1))).strftime("%Y-%m-%d")
        self.dataset = self.dataset[self.dataset['date']<=end_date]
        self.dataset['value_real'] = self.dataset['value']
        self.all_sparse_cols_size = self.dataset[self.all_sparse_feature_cols].max().values.astype(np.int32) + 1
        scaler = StandardScaler()
        self.dataset[self.all_dense_feature_cols] = scaler.fit_transform(self.dataset[self.all_dense_feature_cols])

        self.models = self.create_model()
        merged_res_df = pd.DataFrame()
        for _bin in range(len(self.rank_bins) - 1):
            bin_dataset = self.dataset[
                (self.dataset['rank'] > self.rank_bins[_bin]) & (self.dataset['rank'] <= self.rank_bins[_bin + 1])].dropna()
            logger.info(
                ">>>>>>>>>>{}th bin {} starts".format(_bin + 1, [self.rank_bins[_bin], self.rank_bins[_bin + 1]]))
            pr = multiprocessing.Pool()
            prs = []
            for id, sdf in bin_dataset.groupby('id'):
                prs.append(pr.apply_async(Utils.encoder_decoder_data_generater, args=(id, sdf, self.validation_days, self.encoder_timesteps, self.decoder_timesteps,
                                                                                      self.encoder_dense_feature_cols, self.encoder_sparse_feature_cols,
                                                                                      self.decoder_dense_feature_cols, self.decoder_sparse_feature_cols,
                                                                                      self.decoder_output_col, self.train_date_gap)))
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
                val_encoder_dense_input_data.extend(data[6])
                val_encoder_sparse_input_data.extend(data[7])
                val_decoder_dense_input_data.extend(data[8])
                val_decoder_sparse_input_data.extend(data[9])
                val_output_data.extend(data[10])
                val_weight_data.extend(data[11])
#                predict_encoder_dense_input_data.extend(data[12])
#                predict_encoder_sparse_input_data.extend(data[13])
#                predict_decoder_dense_input_data.extend(data[14])
#                predict_decoder_sparse_input_data.extend(data[15])
#
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
                ## predict results
                item_df = self.dataset[self.dataset['date'] == self.args.cur_date][['rank', 'id', 'name']].reset_index(
                    drop=True)
                nextdate_thelast = (
                        datetime.datetime.strptime(self.args.cur_date, "%Y-%m-%d") + datetime.timedelta(
                    days=self.args.forecast_days - 1)).strftime(
                    "%Y-%m-%d")
                date_df = pd.DataFrame(
                    pd.date_range(self.args.cur_date, nextdate_thelast, freq='D').strftime("%Y-%m-%d").tolist(),
                    columns=['date'])
                item_df['key'] = 0
                date_df['key'] = 0
                res_df = pd.merge(item_df, date_df, how='left', on='key')
                res_df.sort_values(by=['id', 'rank', 'date'], ascending=[True, True, True], inplace=True)
                res_df.drop(['key', 'rank'], axis=1, inplace=True)

                ## predict
                predict_encoder_dense_input_data = []
                predict_encoder_sparse_input_data = []
                predict_decoder_dense_input_data = []
                predict_decoder_sparse_input_data = []

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
                    predict_decoder_dense_input_data.append(
                        [sdf.iloc[sdf.shape[0] - self.decoder_timesteps + j][self.decoder_dense_feature_cols].values.astype(
                            np.float32) for j in
                         range(self.decoder_timesteps)])
                    predict_decoder_sparse_input_data.append(
                        [sdf.iloc[sdf.shape[0] - self.decoder_timesteps + j][self.decoder_sparse_feature_cols].values.astype(
                            np.float32) for j
                         in range(self.decoder_timesteps)])

                predict_encoder_dense_input_data = np.array(predict_encoder_dense_input_data)
                predict_encoder_sparse_input_data = np.array(predict_encoder_sparse_input_data)
                predict_decoder_dense_input_data = np.array(predict_decoder_dense_input_data)
                predict_decoder_sparse_input_data = np.array(predict_decoder_sparse_input_data)

                res_df['value'] = list(
                    map(lambda x: x - 1 if x > 1 else 0, self.predict_model(_bin, [predict_encoder_dense_input_data, predict_encoder_sparse_input_data, predict_decoder_dense_input_data, predict_decoder_sparse_input_data])))
                res_df['value'] = res_df['value'].astype('int')

                merged_res_df = merged_res_df.append(res_df)

                self.res_df = merged_res_df

    def predict_model(self, _bin, predict_x):
        return self.models[_bin].predict(predict_x).reshape(-1).tolist()

    def fit_model(self, _bin, train_x, train_y, train_w, val_x, val_y, val_w):
        logger.info(
            ">>>>>>>>>>{}th bin model params:".format(_bin + 1, self.model_params))
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0.0001, verbose=1)
        earlystopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=8, verbose=1, mode='min',
                                      restore_best_weights=True)
        self.models[_bin].fit(train_x, train_y, sample_weight=train_w
                              , epochs=100
                              , batch_size=self.model_params['batch_size']
                              , validation_data=(val_x, val_y, val_w)
                              , callbacks=[reduce_lr, earlystopping])

    def create_lstm_model(self):
        ##input
        encoder_dense_input = Input(shape=(self.encoder_timesteps, len(self.encoder_dense_feature_cols)))
        encoder_sparse_input = Input(shape=(self.encoder_timesteps, len(self.encoder_sparse_feature_cols)))
        decoder_dense_input = Input(shape=(self.decoder_timesteps, len(self.decoder_dense_feature_cols)))
        decoder_sparse_input = Input(shape=(self.decoder_timesteps, len(self.decoder_sparse_feature_cols)))

        ##embedding
        emds = [Embedding(self.all_sparse_cols_size[i], self.model_params['embedding_size'],
                          embeddings_regularizer=tf.keras.regularizers.l2(self.model_params['l2_reg'])) for i in
                range(len(self.all_sparse_feature_cols))]

        encoder_emd = (Concatenate(axis=2)([emds[self.encoder_sparse_feature_index[i]](
            encoder_sparse_input[:, :, i]) for i in range(len(self.encoder_sparse_feature_cols))]))
        decoder_emd = (Concatenate(axis=2)([emds[self.decoder_sparse_feature_index[i]](
            decoder_sparse_input[:, :, i]) for i in range(len(self.decoder_sparse_feature_cols))]))

        encoder_input = Concatenate(axis=2)([encoder_dense_input, encoder_emd])
        decoder_input = Concatenate(axis=2)([decoder_dense_input, decoder_emd])

        ##encoder
        encoder_output, state_h, state_c = LSTM(self.model_params['latent_dim'], dropout=self.model_params['dropout'],
                                                return_sequences=True, return_state=True)(encoder_input)
        encoder_states = [state_h, state_c]
        ##decoder
        decoder_output, _, _ = LSTM(self.model_params['latent_dim'], dropout=self.model_params['dropout'],
                                    return_sequences=True, return_state=True)(decoder_input,
                                                                              initial_state=encoder_states)
        ##output
        output = decoder_output
        for num in self.model_params['dense_dims']:
            output = Dense(num, kernel_regularizer=tf.keras.regularizers.l2(self.model_params['l2_reg']))(output)
            output = BatchNormalization()(output)
            output = Activation('relu')(output)
            output = Dropout(self.model_params['dropout'])(output)

        output = Dense(1, name='output')(output)

        lstm_model = tf.keras.models.Model([encoder_dense_input, encoder_sparse_input, decoder_dense_input, decoder_sparse_input],
                           output)
        lstm_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.model_params['learning_rate']),
                           loss=tf.keras.losses.MeanSquaredError(),
                           metrics=['MSE', 'MAE', 'MAPE']
                           )

        return lstm_model

    def create_model(self):
        return [self.create_lstm_model() for i in range(len(self.rank_bins) - 1)]


if __name__ == "__main__":
    debug_args = ["--model_class", "LSTMModel"
        , "--model_name", "lstm_v1.0"
        , "--cur_date", "2022-11-10"
        , "--dataset_table", "ods_actttdsearchdb.adm_srh_algo_group_tour_uv_forecast_train_dataset_for_xgb_v2"
        , "--export_table", "ods_actttdsearchdb.adm_srh_algo_group_tour_line_uv_forecast_model_results"
        , "--params_table", "ods_actttdsearchdb.adm_srh_algo_group_tour_line_uv_forecast_model_best_params"
        , "--sparse_feature_cols", "date_type", "holiday_type"
        , "--normal_feature_cols", "covid19", "predate_click_uv", "pre2date_click_uv", "pre3date_click_uv",
                  "pre4date_click_uv", "pre5date_click_uv", "pre6date_click_uv", "pre7date_click_uv",
                  "avg_preweek_click_uv", "avg_premonth_click_uv", "avg_preyear_click_uv", "avg_dod_diff",
                  "avg_wow_diff", "avg_mom_diff", "avg_yoy_diff", "avg_dod_ratio", "avg_wow_ratio", "avg_mom_ratio",
                  "avg_yoy_ratio"
        , "--shift_feature_cols", "date_type", "holiday_type", "day_of_week", "week_of_year", "next_holiday_datediff",
                  "last_holiday_datediff", "next_normalday_datediff", "last_normalday_datediff"
        , "--encoder_dense_feature_cols", 'value', 'predate_click_uv', 'pre2date_click_uv', 'pre3date_click_uv',
                  'pre4date_click_uv', 'pre5date_click_uv',
                  'pre6date_click_uv', 'pre7date_click_uv', 'avg_preweek_click_uv', 'avg_premonth_click_uv',
                  'avg_preyear_click_uv',
                  'avg_dod_diff', 'avg_wow_diff', 'avg_mom_diff', 'avg_yoy_diff', 'avg_dod_ratio', 'avg_wow_ratio',
                  'avg_mom_ratio',
                  'avg_yoy_ratio', 'next_holiday_datediff', 'last_holiday_datediff', 'next_normalday_datediff',
                  'last_normalday_datediff'
        , "--encoder_sparse_feature_cols", 'rank', 'date_type', 'holiday_type', 'day_of_week', 'week_of_year'
        , "--decoder_dense_feature_cols", 'next_holiday_datediff', 'last_holiday_datediff', 'next_normalday_datediff',
                  'last_normalday_datediff'
        , "--decoder_sparse_feature_cols", 'rank', 'date_type', 'holiday_type', 'day_of_week', 'week_of_year'
        , "--target_col", "click_uv"
        , "--date_col", "uv_date"
        , "--id_col", "productlineid"
        , "--name_col", "productlinename"
        , "--rank_col", "rk"
        , "--rank_limit", "10"
        , "--forecast_days", "5"
        , "--validation_days", "2"
        , "--custom_params", ''
        , "--model_load", "False"
        , "--model_tune", 'False'
        , "--model_predict", 'True'
                  ]
    pwd = os.environ['PWD'] if os.environ['PWD'] else ''
    logger.info('pwd:{}'.format(pwd))
    mode = 'debug' if pwd == '/home/powerop/work' else 'run'

    args = Utils.get_args(mode=mode, debug_args=debug_args)
    Model = getattr(sys.modules[__name__], args.model_class)
    model = Model(args)
    model.pipline()

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
import logging
import pyspark.sql.functions as F
from pyspark.sql.window import Window
from pyspark.sql import SparkSession
import sys
import os
import json
import gc
import re
import math
from utils.data_processing import DataReader

os.environ["PYSPARK_PYTHON"] = "/usr/bin/python3.6.5"

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stderr, level=logging.INFO, format=
'''[%(levelname)s] [%(asctime)s] [%(threadName)s] [%(name)s] '''
'''[%(filename)s:%(funcName)s:%(lineno)d]: %(message)s''')

class BaseModel:
    def __init__(self, args):
        self.args = args
        logger.info(">>>>>>>>>>args:{}:".format(self.args))
        if self.args.data_pattern=='hive':
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

        #        all_cols = list(
        #            set(self.args.sparse_feature_cols + self.args.normal_feature_cols + self.args.shift_feature_cols + self.args.encoder_dense_feature_cols + self.args.encoder_sparse_feature_cols + self.args.decoder_dense_feature_cols + self.args.decoder_sparse_feature_cols + [
        #                self.args.target_col] + [self.args.date_col] + [self.args.id_col] + [self.args.name_col] + [
        #                    self.args.rank_col]).intersection(set(dataset_sdf.columns)))

        ## all columns
        all_cols = list(
            set(self.args.sparse_feature_cols + self.args.normal_feature_cols + self.args.shift_feature_cols + self.args.encoder_dense_feature_cols + self.args.encoder_sparse_feature_cols + self.args.decoder_dense_feature_cols + self.args.decoder_sparse_feature_cols + [
                self.args.target_col] + [self.args.date_col] + [self.args.id_col] + [self.args.name_col] + [
                    self.args.rank_col]))
        logger.info(">>>>>>>>>>All columns: {}".format(all_cols))

        ## rank limit
        rank_lower_limit = 0
        rank_upper_limit = float("inf")
        if self.args.rank_bins:
            rank_lower_limit = min(self.args.rank_bins)
            rank_upper_limit = max(self.args.rank_bins)
        elif self.args.rank_limit:
            rank_upper_limit = self.args.rank_limit
        logger.info(">>>>>>>>>>Rank limits from {} to {}".format(rank_lower_limit, rank_upper_limit))

        ## read data
        if self.args.data_pattern == 'csv':
            dataset = pd.read_csv(self.args.data_path)
            self.data_date = dataset['d'][0]
            dataset = dataset[
                (dataset[self.args.rank_col] <= rank_upper_limit) & (dataset[self.args.rank_col] >= rank_lower_limit)][
                all_cols]

        elif self.args.data_pattern == 'hive':
            if self.args.data_path.split('.')[0][:3] == 'tmp':
                dataset_sdf = self.spark.table(self.args.data_path)
            else:
                if 'data_date' in self.custom_params.keys():
                    self.data_date = self.custom_params['data_date']
                else:
                    self.data_date = \
                    self.spark.table(self.args.data_path).select(F.col("d")).orderBy(F.col("d").desc()).limit(
                        1).toPandas().iloc[0][0]
                dataset_sdf = self.spark.table(self.args.data_path).where(F.col('d') == self.data_date)
            dataset_sdf = dataset_sdf.select(all_cols).where(F.col(self.args.rank_col) <= rank_upper_limit).where(
                F.col(self.args.rank_col) >= rank_lower_limit)
            dataset_sdf = dataset_sdf.withColumn("index", F.row_number().over(Window().orderBy(F.lit("A")))).persist()
            dataset = DataReader.spark_to_pandas(dataset_sdf, "index", 1000000)
        logger.info(">>>>>>>>>>Hive data date: {}".format(self.data_date))
        ## normalize columns' names
        dataset.rename(columns={self.args.date_col: "date", self.args.id_col: "id", self.args.name_col: "name",
                                self.args.rank_col: "rank", self.args.target_col: "value"}, inplace=True)
        dataset.rename(columns=lambda x: x.replace(self.args.target_col, 'value'), inplace=True)
        self.args.sparse_feature_cols = list(
            map(lambda x: x.replace(self.args.target_col, 'value'), self.args.sparse_feature_cols))
        self.args.normal_feature_cols = list(
            map(lambda x: x.replace(self.args.target_col, 'value'), self.args.normal_feature_cols))
        self.args.shift_feature_cols = list(
            map(lambda x: x.replace(self.args.target_col, 'value'), self.args.shift_feature_cols))
        self.args.encoder_dense_feature_cols = list(
            map(lambda x: x.replace(self.args.target_col, 'value'), self.args.encoder_dense_feature_cols))
        self.args.encoder_sparse_feature_cols = list(
            map(lambda x: x.replace(self.args.target_col, 'value'), self.args.encoder_sparse_feature_cols))
        self.args.decoder_dense_feature_cols = list(
            map(lambda x: x.replace(self.args.target_col, 'value'), self.args.decoder_dense_feature_cols))
        self.args.decoder_sparse_feature_cols = list(
            map(lambda x: x.replace(self.args.target_col, 'value'), self.args.decoder_sparse_feature_cols))

        self.args.sparse_feature_cols = list(
            map(lambda x: x.replace(self.args.rank_col, 'rank'), self.args.sparse_feature_cols))
        self.args.normal_feature_cols = list(
            map(lambda x: x.replace(self.args.rank_col, 'rank'), self.args.normal_feature_cols))
        self.args.shift_feature_cols = list(
            map(lambda x: x.replace(self.args.rank_col, 'rank'), self.args.shift_feature_cols))
        self.args.encoder_dense_feature_cols = list(
            map(lambda x: x.replace(self.args.rank_col, 'rank'), self.args.encoder_dense_feature_cols))
        self.args.encoder_sparse_feature_cols = list(
            map(lambda x: x.replace(self.args.rank_col, 'rank'), self.args.encoder_sparse_feature_cols))
        self.args.decoder_dense_feature_cols = list(
            map(lambda x: x.replace(self.args.rank_col, 'rank'), self.args.decoder_dense_feature_cols))
        self.args.decoder_sparse_feature_cols = list(
            map(lambda x: x.replace(self.args.rank_col, 'rank'), self.args.decoder_sparse_feature_cols))

        if self.args.differential_col is not None:
            self.args.differential_col = self.args.differential_col.replace(self.args.target_col, 'value')
        if 'rank' not in dataset.columns:
            logger.info(">>>>>>>>>>There is no rank colum.")
            dataset['rank'] = 1

        ## dealing with missing data
        for col in self.args.sparse_feature_cols:
            dataset.loc[dataset[col] == self.args.missing_value, [col]] = dataset[col].max() + 1
        for col in self.args.encoder_sparse_feature_cols:
            dataset.loc[dataset[col] == self.args.missing_value, [col]] = dataset[col].max() + 1
        for col in self.args.decoder_sparse_feature_cols:
            dataset.loc[dataset[col] == self.args.missing_value, [col]] = dataset[col].max() + 1

        #dataset['weight'] = dataset.apply(
        #    lambda x: 1000 if x['date_type']==2 and x['date']>='2021-01-01' else 1, axis=1)
        dataset['weight'] = dataset.apply(
            lambda x: max(self.args.weight_minimum,
                          math.pow(math.e,
                                   -0.693147 / self.args.weight_half_life * (
                                           datetime.datetime.strptime(
                                               self.args.cur_date,
                                               "%Y-%m-%d") - datetime.datetime.strptime(
                                       x['date'],
                                       "%Y-%m-%d")).days)
                          ), axis=1)

        dataset.sort_values(by=['date', 'rank', 'id'], ascending=[True, True, True], inplace=True)
        self.max_value = dataset['value'].max()
        self.dataset = dataset
        self.rank_bins = self.get_rank_bins()
        logger.info(">>>>>>>>>>rank_bins:{}:".format(self.rank_bins))

        if self.args.model_load:
            base_model_name = re.search(r'^(.*?)(_test.*?)?$', self.args.model_name, re.M | re.I).group(1)
            params_d = self.spark.table(self.args.params_path).where(F.col('d') < self.args.cur_date).where(
                F.col('model') == base_model_name).select(
                F.col("d")).orderBy(
                F.col("d").desc()).limit(1).toPandas()

            if params_d.shape[0] > 0:
                max_d = params_d.iloc[0][0]
                self.loaded_params = self.spark.table(self.args.params_path).where(F.col('d') == max_d).where(
                    F.col('model') == base_model_name).toPandas()
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

            now_dataset = now_dataset.dropna()

            ## train
            val_split_date = (datetime.datetime.strptime(self.args.cur_date, "%Y-%m-%d") + datetime.timedelta(
                days=-(self.args.validation_days + self.args.forecast_days))).strftime("%Y-%m-%d")
            val_end_date = (datetime.datetime.strptime(self.args.cur_date, "%Y-%m-%d") + datetime.timedelta(
                days=-(self.args.forecast_days))).strftime("%Y-%m-%d")
            logger.info(">>>>>>>>>>val_split_date:{}".format(val_split_date))
            logger.info(">>>>>>>>>>val_end_date:{}".format(val_end_date))

            now_train_dataset = now_dataset[now_dataset['date'] < val_split_date]
            now_train_y = now_train_dataset[['value_' + str(_day) for _day in range(self.args.forecast_days)]].values
            now_train_x = now_train_dataset.drop(
                ['id', 'date', 'weight'] + ['value_' + str(_day) for _day in range(self.args.forecast_days)],
                axis=1).values
            now_train_w = now_train_dataset['weight'].values

            now_val_dataset = now_dataset[
                (now_dataset['date'] >= val_split_date) & (now_dataset['date'] <= val_end_date)]
            now_val_y = now_val_dataset[['value_' + str(_day) for _day in range(self.args.forecast_days)]].values
            now_val_x = now_val_dataset.drop(
                ['id', 'date', 'weight'] + ['value_' + str(_day) for _day in range(self.args.forecast_days)],
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

                res_df['forecast_value'] = list(
                    map(lambda x: x - 1 if x > 1 else 0, self.predict_model(_bin, 0, now_predict_x)))
                res_df['forecast_value'] = res_df['forecast_value'].astype('int')

                res_df['true_value'] = pd.merge(res_df, self.dataset, how='left', on=['id','date'])['value']

                res_df['true_value'] = res_df.apply(lambda x: x['true_value'] if x['date']<self.data_date else -1, axis=1)

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
                    self.args.normal_feature_cols + predict_feature_cols + ['id', 'date']]

                shift_df = bin_dataset.groupby(['id'])[self.args.shift_feature_cols + ['value', 'weight']].shift(
                    -_day)


                now_dataset = pd.concat([normal_feature_df, shift_df], axis=1)

                if 'avg_yoy_2w_ratio_v3' in now_dataset.columns:
                    now_dataset['yoy_2w_forecast_value'] = now_dataset['avg_yoy_2w_ratio_v3'] * now_dataset[
                        'avg_preyear_d_value_v3']
                elif 'avg_yoy_2w_ratio_align' in now_dataset.columns:
                    now_dataset['yoy_2w_forecast_value'] = now_dataset['avg_yoy_2w_ratio_align'] * now_dataset[
                        'avg_preyear_d_value_align']

                ## differential
                if self.args.differential_col is not None:
                    if self.args.differential_col[-2:] == '_d':
                        now_dataset['value'] = now_dataset['value'] - now_dataset[
                            (self.args.differential_col[:-1] + str(_day + 1) + 'd')]
                    else:
                        now_dataset['value'] = now_dataset['value'] - now_dataset[self.args.differential_col]

                now_dataset = now_dataset.dropna()
                if 'holiday_type' in bin_dataset.columns:
                    now_dataset['holiday_type'] = now_dataset['holiday_type'].astype(int)
                now_dataset = pd.get_dummies(now_dataset, columns=self.args.sparse_feature_cols)
                now_dataset['shift_date'] = list(map(lambda x: (
                            datetime.datetime.strptime(x, "%Y-%m-%d") + datetime.timedelta(days=_day * 30)).strftime(
                    "%Y-%m-%d"), now_dataset['date']))

                ## train
                val_split_date = (datetime.datetime.strptime(self.args.cur_date, "%Y-%m-%d") + datetime.timedelta(
                    days=-(self.args.validation_days + _day))).strftime("%Y-%m-%d")
                val_end_date = (datetime.datetime.strptime(self.args.cur_date, "%Y-%m-%d") + datetime.timedelta(
                    days=-_day)).strftime("%Y-%m-%d")
                logger.info(">>>>>>>>>>val_split_date:{}".format(val_split_date))
                logger.info(">>>>>>>>>>val_end_date:{}".format(val_end_date))

                now_train_dataset = now_dataset[now_dataset['date'] < val_split_date]
                ## 去掉疫情高峰期
                if self.args.drop_covid19==1:
                    now_train_dataset = now_train_dataset.query(
                        "not(shift_date>='2020-01-01' and shift_date<='2020-05-31') and not(shift_date>='2021-08-01' and shift_date<='2021-08-31') and not(shift_date>='2022-03-01' and shift_date<='2022-05-31')")
                ## 去掉2020,2021,2022疫情三年
                elif self.args.drop_covid19==2:
                    now_train_dataset = now_train_dataset.query(
                        "not(shift_date>='2020-01-01' and shift_date<='2022-11-30')")
                ## 去掉2020,2022两年
                elif self.args.drop_covid19==3:
                    now_train_dataset = now_train_dataset.query(
                        "not(shift_date>='2020-01-01' and shift_date<='2022-11-30') or (shift_date>='2021-01-01' and shift_date<='2021-12-31')")

                now_train_y = now_train_dataset['value'].values
                now_train_x = now_train_dataset.drop(['id', 'date', 'weight', 'value', 'shift_date'], axis=1).values
                now_train_w = now_train_dataset['weight'].values

                now_val_dataset = now_dataset[
                    (now_dataset['date'] >= val_split_date) & (now_dataset['date'] < val_end_date)]
                now_val_y = now_val_dataset['value'].values
                now_val_x = now_val_dataset.drop(['id', 'date', 'weight', 'value', 'shift_date'], axis=1).values
                now_val_w = now_val_dataset['weight'].values

                self.x_num = now_train_x.shape[1]

                if self.args.model_tune:
                    self.tune_model(_bin, _day, now_train_x, now_train_y, now_train_w, now_val_x, now_val_y, now_val_w)

                self.fit_model(_bin, _day, now_train_x, now_train_y, now_train_w, now_val_x, now_val_y, now_val_w)
                ## predict
                now_predict_dataset = now_dataset[now_dataset['date'] <= self.args.cur_date]
                now_predict_x = now_predict_dataset.drop(['id', 'date', 'weight', 'value', 'shift_date'], axis=1).values

                #                now_predict_dataset[predict_value_col] = list(
                #                    map(lambda x: x if x > 1 else 1, list(self.predict_model(_bin, _day, now_predict_x))))

                now_predict_dataset[predict_value_col] = list(self.predict_model(_bin, _day, now_predict_x))

                ## differential
                if self.args.differential_col is not None:
                    logger.info(">>>>>>>>>>The differential applied!")
                    if self.args.differential_col[-2:] == '_d':
                        now_predict_dataset[predict_value_col] = now_predict_dataset[predict_value_col] + \
                                                                 now_predict_dataset[(
                                                                             self.args.differential_col[:-1] + str(
                                                                         _day + 1) + 'd')]
                        now_predict_dataset[predict_value_col] = now_predict_dataset.apply(
                            lambda x: x[predict_value_col] if x[predict_value_col] > 1 else 1, axis=1)
                    else:
                        now_predict_dataset[predict_value_col] = now_predict_dataset[predict_value_col] + \
                                                                 now_predict_dataset[self.args.differential_col]
                        now_predict_dataset[predict_value_col] = now_predict_dataset.apply(
                            lambda x: x[predict_value_col] if x[predict_value_col] > 1 else 1, axis=1)

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
            res_df['forecast_value'] = self.dataset[self.dataset['date'] == self.args.cur_date][
                predict_feature_cols].values.reshape(
                [-1, 1])
            res_df['forecast_value'] = res_df['forecast_value'] - 1

            res_df['true_value'] = pd.merge(res_df, self.dataset, how='left', on=['id', 'date'])['value']
            res_df['true_value'] = res_df.apply(lambda x: x['true_value'] if x['date'] < self.data_date else -1, axis=1)

            self.res_df = res_df

    def save_data(self):
        logger.info(">>>>>>>>>>Start saving data:")
        if self.res_df is None:
            raise Exception(">>>>>>>>>>There is no predict results!")
        if not self.args.export_true_value:
            self.res_df = self.res_df[['id','name','date','forecast_value']]

        if self.args.data_pattern == 'csv':
            self.res_df.to_csv(self.args.export_path)
        elif self.args.data_pattern == 'hive':
            spark_df = self.spark.createDataFrame(self.res_df)
            spark_df.createOrReplaceTempView('table_temp')
#            pre_date = (datetime.datetime.strptime(self.args.cur_date, "%Y-%m-%d") + datetime.timedelta(days=-1)).strftime(
#                "%Y-%m-%d")
            if self.args.export_path.split('.')[0][:3] == 'tmp':
                if self.args.export_true_value:
                    self.spark.sql("CREATE TABLE IF NOT EXISTS {0} (id String, name STRING, date STRING, forecast_value DOUBLE, true_value DOUBLE) PARTITIONED BY (d STRING, model STRING)".format(self.args.export_path))
                else:
                    self.spark.sql("CREATE TABLE IF NOT EXISTS {0} (id String, name STRING, date STRING, forecast_value DOUBLE) PARTITIONED BY (d STRING, model STRING)".format(self.args.export_path))

                self.spark.sql("INSERT OVERWRITE TABLE {0} PARTITION (d='{1}', model='{2}') SELECT * FROM table_temp".format(self.args.export_path, self.args.cur_date, self.args.model_name))
            else:
                if self.args.export_mode == 'insert':
                    self.spark.sql(
                        "INSERT INTO table {0} PARTITION (d='{1}', model='{2}') SELECT * FROM table_temp".format(
                            self.args.export_path, self.args.cur_date, self.args.model_name))
                elif self.args.export_mode == 'overwrite':
                    self.spark.sql(
                        "INSERT OVERWRITE table {0} PARTITION (d='{1}', model='{2}') SELECT * FROM table_temp".format(
                            self.args.export_path, self.args.cur_date, self.args.model_name))
                else:
                    logger.info(">>>>>>>>>>Illegal export mode!")
        else:
            raise Exception(">>>>>>>>>>Wrong data pattern !")

    def save_params(self):
        logger.info(">>>>>>>>>>Start saving model params:")
        best_params_df = pd.DataFrame(self.best_params, columns=['bin_number', 'model_number', 'best_params'])
        spark_df = self.spark.createDataFrame(best_params_df)
        spark_df.createOrReplaceTempView('table_temp')
        pre_date = (datetime.datetime.strptime(self.args.cur_date, "%Y-%m-%d") + datetime.timedelta(days=-1)).strftime(
            "%Y-%m-%d")
        if self.args.export_path.split('.')[0][:3] == 'tmp':
            self.spark.sql("DROP TABLE IF EXISTS {0} ".format(self.args.params_path))
            self.spark.sql("CREATE TABLE {0} SELECT * FROM table_temp".format(self.args.params_path))
        else:
            self.spark.sql(
                "INSERT OVERWRITE table {0} PARTITION (d='{1}', model='{2}')SELECT * FROM table_temp".format(
                    self.args.params_path, pre_date, self.args.model_name))

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
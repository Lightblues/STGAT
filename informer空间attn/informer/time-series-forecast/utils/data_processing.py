"""
@Time : 2023/2/1 18:01
@Author : mcxing
@File : data_processing.py
@Software: PyCharm
"""

import argparse
import numpy as np
import sys
import math
import pyspark.sql.functions as F
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stderr, level=logging.INFO, format=
'''[%(levelname)s] [%(asctime)s] [%(threadName)s] [%(name)s] '''
'''[%(filename)s:%(funcName)s:%(lineno)d]: %(message)s''')

class DataReader:

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
    def get_args(mode="run", debug_args={}):
        logger.info(">>>>>>>>>>Start reading configurations:")
        parser = argparse.ArgumentParser()
        parser.add_argument("--model_class", type=str, required=True, help='模型类')
        parser.add_argument("--model_name", type=str, required=True, help='模型名称')
        parser.add_argument("--model_type", type=str, required=False, default=None, help='模型类型：single_model-单模型，multi_model-多模型')
        parser.add_argument("--output_type", type=str, required=False, default=None, help='模型输出类型：single_output-单输出，multi_model-多输出')
        parser.add_argument("--multioutput_type", type=str, required=False, default='RegressorChain', help='多模型类型：RegressorChain-链式')
        parser.add_argument("--cur_date", type=str, required=True, help='当天日期')
        parser.add_argument("--data_pattern", type=str, required=False, default='hive', help='数据源模式：hive,csv')
        parser.add_argument("--data_path", type=str, required=False, default='', help='数据源地址')
        parser.add_argument("--export_path", type=str, required=False, default='', help='输出地址')
        parser.add_argument("--export_true_value", type=DataReader.str2bool, required=False, default=False, help="是否输出真实值")
        parser.add_argument("--export_mode", type=str, required=False, default='overwrite', help='输出模型：insert-插入，overwrite-覆盖')
        parser.add_argument("--params_path", type=str, required=False, default='', help='模型参数地址')
        parser.add_argument("--sparse_feature_cols", type=str, nargs='+', required=False, default=[], help='离散特征')
        parser.add_argument("--normal_feature_cols", type=str, nargs='+', required=False, default=[], help='常规特征')
        parser.add_argument("--shift_feature_cols", type=str, nargs='+', required=False, default=[], help='位移特征：多模型时位移预测当天特征')
        parser.add_argument("--encoder_sparse_feature_cols", type=str, nargs='+', required=False, default=[], help='encoder离散特征')
        parser.add_argument("--encoder_dense_feature_cols", type=str, nargs='+', required=False, default=[], help='encoder稠密特征')
        parser.add_argument("--decoder_sparse_feature_cols", type=str, nargs='+', required=False, default=[], help='decoder离散特征')
        parser.add_argument("--decoder_dense_feature_cols", type=str, nargs='+', required=False, default=[], help='decoder稠密特征')
        parser.add_argument("--date_col", type=str, required=False, default='date', help='日期列')
        parser.add_argument("--id_col", type=str, required=False, default='id', help='id列')
        parser.add_argument("--name_col", type=str, required=False, default='name', help='名称列')
        parser.add_argument("--rank_col", type=str, required=False, default='rk', help='排名列')
        parser.add_argument("--rank_limit", type=int, required=False, default=0, help='排名限制')
        parser.add_argument("--rank_bin_length", type=int, required=False, default=0, help='排名bin长度')
        parser.add_argument("--rank_bins", type=int, nargs='+', required=False, default=[], help='排名bins（用于大数据集拆分）')
        parser.add_argument("--target_col", type=str, required=True, help='target列')
        parser.add_argument("--differential_col", type=str, required=False, default=None, help='差分列')
        parser.add_argument("--weight_half_life", type=int, required=False, default=30, help='时间衰减半衰期')
        parser.add_argument("--weight_minimum", type=float, required=False, default=1, help='时间衰减最小值')
        parser.add_argument("--forecast_days", type=int, required=False, default=30, help='预测天数')
        parser.add_argument("--validation_days", type=int, required=False, default=0, help='验证集天数')
        parser.add_argument("--huber_slope_quantile", type=float, required=False, default=1, help='huber_loss超参分位数')
        parser.add_argument("--drop_covid19", type=int, required=False, default=0, help='去除疫情期间数据：0-不去除，1-去除疫情高峰期，2-去除2020-2022三年，3-去除2020,2022两年')
        parser.add_argument("--missing_value", type=int, required=False, default=-99999, help='缺失值的标识值')
        parser.add_argument("--custom_params", type=str, required=False, default={}, help='自定义参数')
        parser.add_argument("--model_load", type=DataReader.str2bool, required=False, default=False, help='是否加载调参参数（xgb模型）')
        parser.add_argument("--model_tune", type=DataReader.str2bool, required=False, default=False, help='是否调参（xgb模型）')
        parser.add_argument("--params_to_tune", type=str, nargs='+', required=False, default=[], help='调参参数（xgb模型）')
        parser.add_argument("--candidate_params", type=str, required=False, default={}, help='调参候选集（xgb模型）')
        parser.add_argument("--model_predict", type=DataReader.str2bool, required=False, default=True, help='是否预测')

        if mode == 'debug':
            return parser.parse_args(debug_args)
        return parser.parse_args()

class DataGenerater:
    @staticmethod
    def encoder_decoder_data_generater(id, sdf, validation_days, encoder_timesteps, decoder_timesteps,
                                       encoder_dense_feature_cols, encoder_sparse_feature_cols,
                                       decoder_dense_feature_cols, decoder_sparse_feature_cols,
                                       decoder_output_col, train_date_gap=1, differential_col=None, start_token_len=0):

        sdf.reset_index(drop=True, inplace=True)
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

        fsdf = sdf[sdf['date']<='2019-12-31']
        for i in range(fsdf.shape[0] - encoder_timesteps - 2 * decoder_timesteps - validation_days, 0,
                       -train_date_gap):
            train_encoder_dense_input_data.append(
                [fsdf.iloc[i + j][encoder_dense_feature_cols].values.astype(np.float32) for j in
                 range(encoder_timesteps)])
            train_encoder_sparse_input_data.append(
                [fsdf.iloc[i + j][encoder_sparse_feature_cols].values.astype(np.float32) for j in
                 range(encoder_timesteps)])
            train_decoder_dense_input_data.append([np.append(fsdf.iloc[i+encoder_timesteps-1][['value_advance_{}d'.format(j-start_token_len+1)]].values if 'value_advance_{}d'.format(j-start_token_len+1) in fsdf.columns else [0], values=fsdf.iloc[i + j + encoder_timesteps - start_token_len][
                decoder_dense_feature_cols].values).astype(np.float32) if j >= start_token_len else
                fsdf.iloc[i + j + encoder_timesteps - start_token_len][
                ['value'] + decoder_dense_feature_cols].values.astype(
                np.float32) for
                j in range(decoder_timesteps + start_token_len)])
            train_decoder_sparse_input_data.append(
                [fsdf.iloc[i + j + encoder_timesteps - start_token_len][decoder_sparse_feature_cols].values.astype(
                    np.float32)
                    for j in range(decoder_timesteps + start_token_len)])
            #            train_output_data.append(
            #                [fsdf.iloc[i + j + encoder_timesteps][decoder_output_col].values.astype(np.float32) for j in
            #                 range(decoder_timesteps)])
            # differential
            if differential_col is not None:
                train_output_data.append(
                    [fsdf.iloc[i + j + encoder_timesteps][decoder_output_col].values.astype(np.float32) -
                     fsdf.iloc[i + encoder_timesteps - 1][decoder_output_col].values.astype(np.float32) for j in
                     range(decoder_timesteps)])
            else:
                train_output_data.append(
                    [fsdf.iloc[i + j + encoder_timesteps][decoder_output_col].values.astype(np.float32) for j in
                     range(decoder_timesteps)])

            train_weight_data.append(
                fsdf.iloc[i + encoder_timesteps + decoder_timesteps - 1]['weight'].astype(np.float32))

            train_encoder_pos_data.append(
                [[j] for j in range(encoder_timesteps)])
            train_decoder_pos_data.append(
                [[j] for j in range(encoder_timesteps-start_token_len, encoder_timesteps + decoder_timesteps)])

         
        if sdf[sdf['date']>='2022-12-01'].shape[0] - encoder_timesteps - 2 * decoder_timesteps - validation_days>0:
            fsdf =sdf[sdf['date']>='2022-12-01']
            for i in range(fsdf.shape[0] - encoder_timesteps - 2 * decoder_timesteps - validation_days, 0,
                       -train_date_gap):
                train_encoder_dense_input_data.append(
                    [fsdf.iloc[i + j][encoder_dense_feature_cols].values.astype(np.float32) for j in
                    range(encoder_timesteps)])
                train_encoder_sparse_input_data.append(
                    [fsdf.iloc[i + j][encoder_sparse_feature_cols].values.astype(np.float32) for j in
                    range(encoder_timesteps)])
                train_decoder_dense_input_data.append([np.append(fsdf.iloc[i+encoder_timesteps-1][['value_advance_{}d'.format(j-start_token_len+1)]].values if 'value_advance_{}d'.format(j-start_token_len+1) in fsdf.columns else [0], values=fsdf.iloc[i + j + encoder_timesteps - start_token_len][
                    decoder_dense_feature_cols].values).astype(np.float32) if j >= start_token_len else
                    fsdf.iloc[i + j + encoder_timesteps - start_token_len][
                    ['value'] + decoder_dense_feature_cols].values.astype(
                    np.float32) for
                    j in range(decoder_timesteps + start_token_len)])
                train_decoder_sparse_input_data.append(
                    [fsdf.iloc[i + j + encoder_timesteps - start_token_len][decoder_sparse_feature_cols].values.astype(
                        np.float32)
                        for j in range(decoder_timesteps + start_token_len)])
                #            train_output_data.append(
                #                [fsdf.iloc[i + j + encoder_timesteps][decoder_output_col].values.astype(np.float32) for j in
                #                 range(decoder_timesteps)])
                # differential
                if differential_col is not None:
                    train_output_data.append(
                        [fsdf.iloc[i + j + encoder_timesteps][decoder_output_col].values.astype(np.float32) -
                        fsdf.iloc[i + encoder_timesteps - 1][decoder_output_col].values.astype(np.float32) for j in
                        range(decoder_timesteps)])
                else:
                    train_output_data.append(
                        [fsdf.iloc[i + j + encoder_timesteps][decoder_output_col].values.astype(np.float32) for j in
                        range(decoder_timesteps)])

                train_weight_data.append(
                    fsdf.iloc[i + encoder_timesteps + decoder_timesteps - 1]['weight'].astype(np.float32))

                train_encoder_pos_data.append(
                    [[j] for j in range(encoder_timesteps)])
                train_decoder_pos_data.append(
                    [[j] for j in range(encoder_timesteps-start_token_len, encoder_timesteps + decoder_timesteps)])


        for i in range(fsdf.shape[0] - encoder_timesteps - 2 * decoder_timesteps,
                       fsdf.shape[0] - encoder_timesteps - 2 * decoder_timesteps - validation_days, -1):
            val_encoder_dense_input_data.append(
                [fsdf.iloc[i + j][encoder_dense_feature_cols].values.astype(np.float32) for j in
                 range(encoder_timesteps)])
            val_encoder_sparse_input_data.append(
                [fsdf.iloc[i + j][encoder_sparse_feature_cols].values.astype(np.float32) for j in
                 range(encoder_timesteps)])
            val_decoder_dense_input_data.append([np.append(fsdf.iloc[i+encoder_timesteps-1][['value_advance_{}d'.format(j-start_token_len+1)]].values if 'value_advance_{}d'.format(j-start_token_len+1) in fsdf.columns else [0], values=fsdf.iloc[i + j + encoder_timesteps - start_token_len][
                decoder_dense_feature_cols].values).astype(np.float32) if j >= start_token_len else
                fsdf.iloc[i + j + encoder_timesteps - start_token_len][
                ['value'] + decoder_dense_feature_cols].values.astype(
                np.float32) for
                j in range(decoder_timesteps + start_token_len)])
            val_decoder_sparse_input_data.append(
                [fsdf.iloc[i + j + encoder_timesteps - start_token_len][decoder_sparse_feature_cols].values.astype(
                    np.float32)
                    for j in range(decoder_timesteps + start_token_len)])

            if differential_col is not None:
                val_output_data.append(
                    [fsdf.iloc[i + j + encoder_timesteps][decoder_output_col].values.astype(np.float32) -
                     fsdf.iloc[i + encoder_timesteps - 1][decoder_output_col].values.astype(np.float32) for j in
                     range(decoder_timesteps)])
            else:
                val_output_data.append(
                    [fsdf.iloc[i + j + encoder_timesteps][decoder_output_col].values.astype(np.float32) for j in
                     range(decoder_timesteps)])
            val_weight_data.append(
                fsdf.iloc[i + encoder_timesteps + decoder_timesteps - 1]['weight'].astype(np.float32))

            val_encoder_pos_data.append(
                [[j] for j in range(encoder_timesteps)])
            val_decoder_pos_data.append(
                [[j] for j in range(encoder_timesteps-start_token_len, encoder_timesteps + decoder_timesteps)])

        i = sdf.shape[0] - encoder_timesteps - decoder_timesteps

        predict_encoder_dense_input_data.append(
            [sdf.iloc[i + j][encoder_dense_feature_cols].values.astype(np.float32) for j in
             range(encoder_timesteps)])
        predict_encoder_sparse_input_data.append(
            [sdf.iloc[i + j][encoder_sparse_feature_cols].values.astype(np.float32) for j in
             range(encoder_timesteps)])
        predict_decoder_dense_input_data.append([np.append(sdf.iloc[i+encoder_timesteps-1][['value_advance_{}d'.format(j-start_token_len+1)]].values if 'value_advance_{}d'.format(j-start_token_len+1) in sdf.columns else [0], values=sdf.iloc[i + j + encoder_timesteps - start_token_len][
            decoder_dense_feature_cols].values).astype(np.float32) if j >= start_token_len else
            sdf.iloc[i + j + encoder_timesteps - start_token_len][
            ['value'] + decoder_dense_feature_cols].values.astype(
            np.float32) for
            j in range(decoder_timesteps + start_token_len)])
        predict_decoder_sparse_input_data.append(
            [sdf.iloc[i + j + encoder_timesteps - start_token_len][decoder_sparse_feature_cols].values.astype(
                np.float32)
                for j in range(decoder_timesteps + start_token_len)])

        predict_encoder_pos_data.append(
            [[j] for j in range(encoder_timesteps)])
        predict_decoder_pos_data.append(
            [[j] for j in range(encoder_timesteps-start_token_len, encoder_timesteps + decoder_timesteps)])

#        logger.info('The data of id{} are done!'.format(id))
        return (train_encoder_dense_input_data, train_encoder_sparse_input_data, train_decoder_dense_input_data,
                train_decoder_sparse_input_data, train_output_data, train_weight_data, train_encoder_pos_data,
                train_decoder_pos_data, val_encoder_dense_input_data, val_encoder_sparse_input_data, val_decoder_dense_input_data,
                val_decoder_sparse_input_data, val_output_data, val_weight_data, val_encoder_pos_data,
                val_decoder_pos_data, predict_encoder_dense_input_data, predict_encoder_sparse_input_data, predict_encoder_pos_data,
                predict_decoder_dense_input_data, predict_decoder_sparse_input_data, predict_decoder_pos_data
                )

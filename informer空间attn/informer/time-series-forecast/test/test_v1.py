"""
@Time : 2023/3/5 16:01
@Author : mcxing
@File : test.py
@Software: PyCharm
"""
import logging
import os
import sys
# sys.path.append("D:/work/code_repository/vacation-ai/time-series-forecast")
sys.path.append("D:/Users/yanlinzhang/Desktop/trip/git/time-series-forecast")

current_path = os.path.dirname(__file__)
parent_path = os.path.dirname(current_path)
sys.path.append(parent_path)
os.environ["PYSPARK_PYTHON"] = "/usr/bin/python3.6.5"

from models.transformer.informer.informer import InformerModel
from models.transformer.STinformer.STinformer import STInformerModel
# from models.xgb.xgb import XgbModel
# from models.mlp.ae_mlp import AEMLPModel
# from models.lstm.lstm import LSTMModel
from utils.data_processing import DataReader

import tensorflow as tf


current_path = os.path.dirname(__file__)
parent_path = os.path.dirname(current_path)
out_path = os.path.dirname(parent_path)
data_path = out_path + "/tsf_data/grp_prov_order_20230101_20231101.csv"
export_path = out_path + "/tsf_data/results/grp_prov_order_20230101_20231101_informer_full.csv"

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stderr, level=logging.WARNING, format=
'''[%(levelname)s] [%(asctime)s] [%(threadName)s] [%(name)s] '''
'''[%(filename)s:%(funcName)s:%(lineno)d]: %(message)s''')

if __name__ == "__main__":
    debug_args = [
        # "--mode", "debug"
        "--model_class", "STInformerModel"
        , "--model_name", "Informer_full"
        , "--cur_date", "2023-07-04"
        , "--data_pattern", "csv"
        #, "--data_path", "D:/work/code_repository/vacation-ai/tsf_data/grp_orders_20170101_20230520.csv"
        , "--data_path", data_path
        #, "--export_path", "D:/work/code_repository/vacation-ai/tsf_data/results/grp_orders_20170101_20230520_0629informer.csv"
        , "--export_path", export_path
        , "--export_true_value", "True"
        , "--params_path", "ods_actttdsearchdb.adm_srh_algo_group_tour_line_uv_forecast_model_best_params"
        , "--export_true_value", "True"
        , "--sparse_feature_cols", "date_type", "holiday_type"
        , "--normal_feature_cols", "avg_preweek_orders", "avg_premonth_orders", "avg_preyear_orders", "avg_dod_diff",
                  "avg_wow_diff", "avg_mom_diff", "avg_dod_ratio", "avg_wow_ratio", "avg_mom_ratio",
                  "orders_advance_1d", "orders_advance_2d", "orders_advance_3d", "orders_advance_4d", "orders_advance_5d", "orders_advance_6d", "orders_advance_7d", "orders_advance_8d", "orders_advance_9d", "orders_advance_10d", 
                  "orders_advance_11d", "orders_advance_12d", "orders_advance_13d", "orders_advance_14d", "orders_advance_15d", "orders_advance_16d", "orders_advance_17d", "orders_advance_18d", "orders_advance_19d", "orders_advance_20d", 
                  "orders_advance_21d", "orders_advance_22d", "orders_advance_23d", "orders_advance_24d", "orders_advance_25d", "orders_advance_26d", "orders_advance_27d", "orders_advance_28d", "orders_advance_29d", "orders_advance_30d"                  
        , "--shift_feature_cols", "date_type", "holiday_type", "day_of_week", "week_of_year", "next_holiday_datediff",
                  "last_holiday_datediff", "next_normalday_datediff", "last_normalday_datediff"
        , "--encoder_dense_feature_cols", 'orders', 'next_holiday_datediff', 'last_holiday_datediff', 'next_normalday_datediff', 'last_normalday_datediff',
                'avg_preyear_m_orders_v3', 'avg_preyear_w_orders_v3', 'avg_preyear_d_orders_v3', 'avg_yoy_m_diff_v3', 
                'avg_yoy_m_ratio_v3', 'avg_yoy_w_diff_v3', 'avg_yoy_w_ratio_v3', 'avg_yoy_d_diff_v3', 'avg_yoy_d_ratio_v3'
        , "--encoder_sparse_feature_cols", 'rk', 'date_type', 'holiday_type', 'day_of_week', 'week_of_year'
        , "--decoder_dense_feature_cols", 'next_holiday_datediff', 'last_holiday_datediff','next_normalday_datediff','last_normalday_datediff',
            'avg_preyear_m_orders_v3', 'avg_preyear_w_orders_v3', 'avg_preyear_d_orders_v3'
        , "--decoder_sparse_feature_cols", 'rk', 'date_type', 'holiday_type', 'day_of_week', 'week_of_year'
        , "--target_col", "orders"
        , "--date_col", "date"
        , "--id_col", "dest_prov_id"
        , "--name_col", "dest_prov_name"
        , "--rank_col", "rk"
        # , "--rank_limit", "3"
        , "--forecast_days", "7"
        , "--validation_days", "7"
        , "--custom_params", '{"train_date_gap":5, "encoder_timesteps":60,  "start_token_len":30, "learning_rate":0.1, "batch_size":8, "l2_reg":0.01, "attn_type":"full", "e_layers":2, "d_layers":2}'
        , "--drop_covid19", "2"
        , "--model_load", "False"
        , "--model_tune", 'False'
        , "--model_predict", 'True'
                  ]
    mode = 'debug'
    if sys.gettrace():
        tf.config.experimental_run_functions_eagerly(True)
    args = DataReader.get_args(mode=mode, debug_args=debug_args)
    Model = getattr(sys.modules[__name__], args.model_class)
    model = Model(args)
    model.pipline()
#    model.read_data()
#    model.train_model()

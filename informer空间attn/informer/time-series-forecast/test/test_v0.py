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
current_path = os.path.dirname(__file__)
parent_path = os.path.dirname(current_path)
out_path = os.path.dirname(parent_path)
data_path = out_path + "\\tsf_data\\grp_suc_20170101_20230822.csv"
export_path = out_path + "\\tsf_data\\results\\grp_suc_20170101_20230822_informer.csv"

sys.path.append(parent_path)
os.environ["PYSPARK_PYTHON"] = "/usr/bin/python3.6.5"
from models.transformer.informer.informer import InformerModel
from utils.data_processing import DataReader



logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stderr, level=logging.WARNING, format=
'''[%(levelname)s] [%(asctime)s] [%(threadName)s] [%(name)s] '''
'''[%(filename)s:%(funcName)s:%(lineno)d]: %(message)s''')

if __name__ == "__main__":
    debug_args = ["--model_class", "InformerModel"
        , "--model_name", "informer_v1.0"
        , "--cur_date", "2023-05-24"
        , "--data_pattern", "csv"
        , "--data_path", data_path
        , "--export_path", export_path
        , "--params_path", "ods_actttdsearchdb.adm_srh_algo_group_tour_line_uv_forecast_model_best_params"
        , "--export_true_value", "True"
        , "--sparse_feature_cols", "date_type", "holiday_type"
        , "--normal_feature_cols", "avg_preweek_suc_np", "avg_premonth_suc_np", "avg_preyear_suc_np", "avg_dod_diff",
                  "avg_wow_diff", "avg_mom_diff", "avg_dod_ratio", "avg_wow_ratio", "avg_mom_ratio",
                  "suc_np_advance_1d", "suc_np_advance_2d", "suc_np_advance_3d", "suc_np_advance_4d", "suc_np_advance_5d", "suc_np_advance_6d", "suc_np_advance_7d", "suc_np_advance_8d", "suc_np_advance_9d", "suc_np_advance_10d", 
                  "suc_np_advance_11d", "suc_np_advance_12d", "suc_np_advance_13d", "suc_np_advance_14d", "suc_np_advance_15d", "suc_np_advance_16d", "suc_np_advance_17d", "suc_np_advance_18d", "suc_np_advance_19d", "suc_np_advance_20d", 
                  "suc_np_advance_21d", "suc_np_advance_22d", "suc_np_advance_23d", "suc_np_advance_24d", "suc_np_advance_25d", "suc_np_advance_26d", "suc_np_advance_27d", "suc_np_advance_28d", "suc_np_advance_29d", "suc_np_advance_30d"                  
        , "--shift_feature_cols", "date_type", "holiday_type", "day_of_week", "week_of_year", "next_holiday_datediff",
                  "last_holiday_datediff", "next_normalday_datediff", "last_normalday_datediff"
        , "--encoder_dense_feature_cols", 'suc_np', 'next_holiday_datediff', 'last_holiday_datediff', 'next_normalday_datediff', 'last_normalday_datediff',
                'avg_preyear_m_suc_np_v3', 'avg_preyear_w_suc_np_v3', 'avg_preyear_d_suc_np_v3', 'avg_yoy_m_diff_v3', 
                'avg_yoy_m_ratio_v3', 'avg_yoy_w_diff_v3', 'avg_yoy_w_ratio_v3', 'avg_yoy_d_diff_v3', 'avg_yoy_d_ratio_v3'
        , "--encoder_sparse_feature_cols", 'rk', 'date_type', 'holiday_type', 'day_of_week', 'week_of_year'
        , "--decoder_dense_feature_cols", 'next_holiday_datediff', 'last_holiday_datediff','next_normalday_datediff','last_normalday_datediff',
            'avg_preyear_m_suc_np_v3', 'avg_preyear_w_suc_np_v3', 'avg_preyear_d_suc_np_v3'
        , "--decoder_sparse_feature_cols", 'rk', 'date_type', 'holiday_type', 'day_of_week', 'week_of_year'
        # suc_np:人头数，为预测目标
        , "--target_col", "suc_np"
        , "--date_col", "date"
        # id列标识符
        , "--id_col", "dep_dest_id"
        , "--name_col", "dep_dest_name"
        # rk：可以理解为和id是一样的东西
        , "--rank_col", "rk"
        # 只取rk前limit个,为了加快调试速度
        , "--rank_limit", "3"
        # 预测天数
        , "--forecast_days", "30"
        # 验证天数,一般取7
        , "--validation_days", "7"
        # 自定义模型参数, 比较重要的是encoder_timesteps, 表示过去的观测序列长度。一般取60
        , "--custom_params", '{"train_date_gap":5, "encoder_timesteps":60,  "start_token_len":30, "learning_rate":0.2, "batch_size":256, "l2_reg":0.01}'
        # 是否去除新冠影响,2为去除
        , "--drop_covid19", "2"
        , "--model_load", "False"
        , "--model_tune", 'False'
        , "--model_predict", 'True'
                  ]
    mode = 'debug'
    args = DataReader.get_args(mode=mode, debug_args=debug_args)
    Model = getattr(sys.modules[__name__], args.model_class)
    model = Model(args)
    model.pipline()
#    model.read_data()
#    model.train_model()

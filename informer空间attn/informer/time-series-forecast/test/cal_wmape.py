import pandas as pd
import numpy as np
from pyspark.sql import SparkSession

def get_wmape(table, d_start, d_end, forecast_date_start, forecast_date_end, models):
    """
    wmpae计算函数
    
    参数：
    table -- 数据表
    d_start -- 分区起始日期
    d_end -- 分区结束日期
    forecast_date_start -- 预测起始日期
    forecast_date_end -- 预测结束日期
    models -- 模型列表
    
    返回值：
    res -- wmape结果
    detail_res --wmape明细结果
    
    """
    # 从数据库中获取数据
    spark = SparkSession.builder.appName("CreateHiveTable").enableHiveSupport().getOrCreate()
    df = spark.sql(f"SELECT model, d, date, id, true_value, forecast_value FROM {table} WHERE d BETWEEN '{d_start}' AND '{d_end}' AND date BETWEEN '{forecast_date_start}' AND '{forecast_date_end}' AND model IN {models} AND true_value <> -1 AND forecast_value <> -1").toPandas()

    # 计算指标
    df['true_value_all_id'] = df.groupby(['model', 'd', 'date'])['true_value'].transform('sum')
    df['forecast_value_all_id'] = df.groupby(['model', 'd', 'date'])['forecast_value'].transform('sum')
    df['true_value_all_date'] = df.groupby(['model', 'd', 'id'])['true_value'].transform('sum')
    df['forecast_value_all_date'] = df.groupby(['model', 'd', 'id'])['forecast_value'].transform('sum')
    df['true_value_all'] = df.groupby(['model', 'd'])['true_value'].transform('sum')
    df['forecast_value_all'] = df.groupby(['model', 'd'])['forecast_value'].transform('sum')

    df['diff_all'] = abs(df['true_value_all'] - df['forecast_value_all'])
    df['diff_all_id'] = abs(df['true_value_all_id'] - df['forecast_value_all_id'])
    df['diff_all_date'] = abs(df['true_value_all_date'] - df['forecast_value_all_date'])
    df['diff'] = abs(df['true_value'] - df['forecast_value'])
    
    wmape_all = df.groupby(['model','d'])['diff_all'].sum()/df.groupby(['model', 'd'])['true_value_all'].sum()
    wmape_all_id = df.groupby(['model','d'])['diff_all_id'].sum()/df.groupby(['model', 'd'])['true_value_all_id'].sum()
    wmape_all_date = df.groupby(['model','d'])['diff_all_date'].sum()/df.groupby(['model', 'd'])['true_value_all_date'].sum()
    wmape = df.groupby(['model','d'])['diff'].sum()/df.groupby(['model', 'd'])['true_value'].sum()

    detail_res = pd.concat([wmape_all, wmape_all_date, wmape_all_id, wmape], axis=1).reset_index()
    detail_res.columns=['model', 'd','wmape_all', 'wmape_all_date', 'wmape_all_id', 'wmape']
    
    res = detail_res.groupby(['model']).agg({
        'wmape_all': 'mean',
        'wmape_all_date': 'mean',        
        'wmape_all_id': 'mean',
        'wmape': 'mean',
        }).reset_index()
    
    return res, detail_res

if __name__ == "__main__":
    
    table = 'tmp_actttdsearchdb.adm_srh_algo_grp_prov_orders_forecast_model_results_tune'
    d_start = '2023-04-11'
    d_end = '2023-04-20'
    forecast_date_start = '2023-04-11'
    forecast_date_end = '2023-04-24'
    models = ('tune_xgb_test','tune_informer_test')
    
    res, detail_res = get_wmape(table, d_start, d_end, forecast_date_start, forecast_date_end, models)
    res.to_csv("wmape.csv")

## 对应hive代码
##SELECT model
##    , avg(wmape_all) AS wmape_all
##    , avg(wmpae_all_id) AS wmpae_all_id
##    , avg(wmpae_all_date) AS wmpae_all_date
##    , avg(wmape) AS wmape
##FROM (
##    SELECT model, d
##        , abs(SUM(true_value) - sum(forecast_value)) / SUM(true_value) AS wmape_all
##        , SUM(abs(true_value_all_id - forecast_value_all_id)) / SUM(true_value_all_id) AS wmpae_all_id
##        , SUM(abs(true_value_all_date - forecast_value_all_date)) / SUM(true_value_all_date) AS wmpae_all_date
##        , SUM(abs(true_value - forecast_value)) / SUM(true_value) AS wmape
##    FROM (
##        SELECT model, d, date, id
##            , true_value
##            , forecast_value
##            , SUM(true_value) OVER (PARTITION BY model, d, date ) AS true_value_all_id
##            , SUM(forecast_value) OVER (PARTITION BY model, d, date ) AS forecast_value_all_id
##            , SUM(true_value) OVER (PARTITION BY model, d, id ) AS true_value_all_date
##            , SUM(forecast_value) OVER (PARTITION BY model, d, id ) AS forecast_value_all_date
##        FROM tmp_actttdsearchdb.adm_srh_algo_grp_prov_orders_forecast_model_results_tune
##        WHERE d BETWEEN '2023-04-11' AND '2023-04-20'
##        AND date BETWEEN '2023-04-11' AND '2023-04-24'
##        AND model IN ('tune_xgb_test', 'tune_informer_test')
##        AND true_value <> -1
##        AND forecast_value <> -1
##    )
##    GROUP BY 1, 2
##)
##GROUP BY 1
##
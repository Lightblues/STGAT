"""
@Time : 2023/3/5 16:01
@Author : mcxing
@File : main.py
@Software: PyCharm
"""
import logging
import os
import sys
from models.transformer.informer.informer import InformerModel
# from models.xgb.xgb import XgbModel
# from models.mlp.ae_mlp import AEMLPModel
# from models.lstm.lstm import LSTMModel
from utils.data_processing import DataReader

# sys.path.append("D:/work/code_repository/vacation-ai/time-series-forecast")
# os.environ["PYSPARK_PYTHON"] = "/usr/bin/python3.6.5"

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stderr, level=logging.WARNING, format=
'''[%(levelname)s] [%(asctime)s] [%(threadName)s] [%(name)s] '''
'''[%(filename)s:%(funcName)s:%(lineno)d]: %(message)s''')

if __name__ == "__main__":
    args = DataReader.get_args()
    Model = getattr(sys.modules[__name__], args.model_class)
    model = Model(args)
    model.pipline()

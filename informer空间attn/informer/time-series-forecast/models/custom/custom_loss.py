"""
@Time : 2023/5/9 17:34
@Author : mcxing
@File : custom_loss.py
@Software: PyCharm
"""

import numpy as np
def get_huber_loss(max_value, huber_slope_quantile):
    def huber_loss(y_true, y_pred):
        # https://en.wikipedia.org/wiki/Huber_loss#Pseudo-Huber_loss_function
        # https://www.cnblogs.com/nxf-rabbit75/p/10440805.html

        error = y_pred - y_true
        scale = 1 + (error / huber_slope) ** 2
        scale_sqrt = np.sqrt(scale)
        g = error / scale_sqrt
        h = 1 / scale / scale_sqrt
        return g, h

    huber_slope = max_value * huber_slope_quantile
    huber_loss.huber_slope = huber_slope
    huber_loss.huber_slope_quantile = huber_slope_quantile

    return huber_loss
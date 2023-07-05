"""
@Time : 2023/5/9 17:34
@Author : mcxing
@File : xgb.py
@Software: PyCharm
"""


import numpy as np
import xgboost
from xgboost.sklearn import XGBRegressor
import json
import gc
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, HalvingGridSearchCV
from models.base.base_model import BaseModel
import logging
import sys
import multiprocessing
from models.custom.custom_loss import get_huber_loss

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stderr, level=logging.INFO, format=
'''[%(levelname)s] [%(asctime)s] [%(threadName)s] [%(name)s] '''
'''[%(filename)s:%(funcName)s:%(lineno)d]: %(message)s''')

class XgbModel(BaseModel):
    def __init__(self, args):
        super(XgbModel, self).__init__(args)

    def create_model(self):
        models = [[XGBRegressor(n_estimators=50,
                                max_depth=11,
                                learning_rate=0.1,
                                verbosity=3,
                                objective=get_huber_loss(self.max_value, self.args.huber_slope_quantile),
                                booster='gbtree',
                                tree_method='hist',
                                gamma=0,
                                min_child_weight=9,
                                max_delta_step=0,
                                subsample=0.8,
                                colsample_bytree=0.8,
                                colsample_bylevel=1,
                                colsample_bynode=1,
                                reg_alpha=1,
                                reg_lambda=1,
                                scale_pos_weight=1,
                                random_state=2022,
                                missing=self.args.missing_value
                                ) for i in range(self.args.forecast_days)] for j in range(len(self.rank_bins) - 1)]

        logger.info(">>>>>>>>>>Default model params: {}".format(models[0][0].get_params))

        for _bin in range(len(self.rank_bins) - 1):
            for _day in range(self.args.forecast_days):
                if self.args.model_load:
                    best_params_list = self.loaded_params[
                        (self.loaded_params['rank_bin_number'].astype(int) == _bin + 1) & (
                                self.loaded_params['model_number'].astype(int) == _day + 1)]['best_params'].values
                    if best_params_list.size > 0:
                        best_params = json.loads(best_params_list[0])
                        if 'huber_slope_quantile' in best_params.keys():
                            best_params['objective'] = get_huber_loss(self.max_value,
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
        o1 = get_huber_loss(self.max_value, 0.8)
        o2 = get_huber_loss(self.max_value, 0.9)
        o3 = get_huber_loss(self.max_value, 0.95)
        o4 = get_huber_loss(self.max_value, 0.99)
        o5 = get_huber_loss(self.max_value, 1)
        
        candidate_params = {'n_estimators': [10, 20, 50, 100, 150, 200, 300, 400],
                            'learning_rate': [0.02, 0.05, 0.1],
                            'max_depth': [5, 7, 9, 11, 13],
                            'min_child_weight': [1, 3, 5, 7, 9, 11],
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
        #        self.model_diagrams[_bin][_day] = tune_params(model=self.model_diagrams[_bin][_day], params=dict(
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
        logger.info(">>>>>>>>>>model params: {}".format(self.models[_bin][_day].get_params))
        logger.info(">>>>>>>>>>Start fit:")
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

            logger.info(">>>>>>>>>>best iteration: {}".format(self.models[_bin][_day].best_iteration))

            if 'train_val' in self.custom_params.keys() and self.custom_params['train_val'] is True:
                logger.info(">>>>>>>>>>Start refit")
                self.models[_bin][_day].set_params(**{'n_estimators': self.models[_bin][_day].best_iteration})
                logger.info(">>>>>>>>>>model refit params: {}".format(self.models[_bin][_day].get_params))
                self.models[_bin][_day].fit(np.append(train_x, values=val_x, axis=0),
                                            np.append(train_y, values=val_y, axis=0),
                                            sample_weight=np.append(train_w, values=val_w, axis=0), verbose=True)
        else:
            self.models[_bin][_day].fit(train_x, train_y, sample_weight=train_w, verbose=True)

    def predict_model(self, _bin, _day, predict_x):
        logger.info(">>>>>>>>>>Start {}th bin {}th model predict:".format(_bin + 1, _day + 1))
        return self.models[_bin][_day].predict(predict_x)
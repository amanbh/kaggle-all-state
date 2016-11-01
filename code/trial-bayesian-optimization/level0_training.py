# -*- coding: utf-8 -*-
"""
@author: Faron, amanbh
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.cross_validation import KFold
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from pylightgbm.models import GBMRegressor
from IPython.display import display
import json
import pickle
import sys

class SklearnWrapper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train, x_valid, y_valid):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)


class XgbWrapper(object):
    def __init__(self, seed=0, params=None):
        self.param = params
        self.param['seed'] = seed
        self.nrounds = params.pop('nrounds', 250)

    def train(self, x_train, y_train, x_valid, y_valid):
        dtrain = xgb.DMatrix(x_train, label=y_train)
        self.gbdt = xgb.train(self.param, dtrain, self.nrounds)

    def predict(self, x):
        return self.gbdt.predict(xgb.DMatrix(x))


class LightgbmWrapper(object):
    def __init__(self, seed=0, params=None, pyLightGBM_managed=True):
        self.params = params
        if pyLightGBM_managed: 
            self.clf = GBMRegressor(**params)
        self.params['pyLightGBM_managed'] = pyLightGBM_managed
        self.params['seed'] = seed

    def train(self, x_train, y_train, x_valid, y_valid):
        if self.params['application'] == "regression":
            self.clf.fit(x_train, np.log1p(y_train), [(x_valid, np.log1p(y_valid))])
        else:
            self.clf.fit(x_train, y_train, [(x_valid, y_valid)])

    def predict(self, x):
        if self.params['application'] == "regression":
            return np.expm1(self.clf.predict(x))
        else:
            return self.clf.predict(x)



def save_results_to_json(name, params, oof_test, oof_train):
    with open(name + '.json', 'w') as outfile:
        json.dump({ 'name': name,
                    'params': params,
                    'oof_train': oof_train.tolist(),
                    'oof_test': oof_test.tolist() },
                  outfile,
                  indent = 4,
                  sort_keys = False)


def load_results_from_json(filename):
    with open(filename, 'r') as infile:
        res = json.load(infile)
        res['oof_train'] = np.array(res['oof_train'])
        res['oof_test']  = np.array(res['oof_test'])
        return res


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

ID = 'id'
TARGET = 'loss'
NFOLDS = 5
SEED = 0
NROWS = None
DATA_DIR = "../../input"

TRAIN_FILE = "{0}/train.csv".format(DATA_DIR)
TEST_FILE = "{0}/test.csv".format(DATA_DIR)
SUBMISSION_FILE = "{0}/sample_submission.csv".format(DATA_DIR)
USE_PICKLED = True

if not USE_PICKLED:
    print("Loading training data from {}".format(TRAIN_FILE))
    train = pd.read_csv(TRAIN_FILE, nrows=NROWS)
    print("Loading test data from {}".format(TEST_FILE))
    test = pd.read_csv(TEST_FILE, nrows=NROWS)
    
    y_train = train[TARGET].ravel()
    
    train.drop([ID, TARGET], axis=1, inplace=True)
    test.drop([ID], axis=1, inplace=True)
    
    print("Data shapes: Train = {}, Test = {}".format(train.shape, test.shape))
    
    ntrain = train.shape[0]
    ntest = test.shape[0]
    train_test = pd.concat((train, test)).reset_index(drop=True)
    
    features = train.columns
    
    cats = [feat for feat in features if 'cat' in feat]
    for feat in cats:
        train_test[feat] = pd.factorize(train_test[feat], sort=True)[0]
    
    print ("Head ( train_test ) : ")
    print (train_test.head())

    x_train = np.array(train_test.iloc[:ntrain,:])
    x_test = np.array(train_test.iloc[ntrain:,:])
    with open('data.pkl', 'wb') as pkl_file:
        pickle.dump( (x_train, x_test, y_train), pkl_file)
else:
    with open('data.pkl', 'rb') as pkl_file:
        (x_train, x_test, y_train) = pickle.load(pkl_file)
        ntrain = x_train.shape[0]
        ntest  = x_test.shape[0]


kf = KFold(ntrain, n_folds=NFOLDS, shuffle=False, random_state=SEED)


class SklearnWrapper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)


class XgbWrapper(object):
    def __init__(self, seed=0, params=None):
        self.param = params
        self.param['seed'] = seed
        self.nrounds = params.pop('nrounds', 250)

    def train(self, x_train, y_train):
        dtrain = xgb.DMatrix(x_train, label=y_train)
        self.gbdt = xgb.train(self.param, dtrain, self.nrounds)

    def predict(self, x):
        return self.gbdt.predict(xgb.DMatrix(x))


class LightgbmWrapper(object):
    def __init__(self, seed=0, params=None):
        self.params = params
        self.clf = GBMRegressor(**params)
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


def get_oof(clf):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]
        y_te = y_train[test_index]

        clf.train(x_tr, y_tr, x_te, y_te)
        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)


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


if False:
    et_params = {
        'n_jobs': 4,
        'n_estimators': 169,
        'max_features': 0.999,
        'max_depth': 12,
        'min_samples_leaf': 12,
        'verbose': 1,
    }
    et = SklearnWrapper(clf=ExtraTreesRegressor, seed=SEED, params=et_params)
    et_oof_train, et_oof_test = get_oof(et)
    save_results_to_json('model_et', et_params, et_oof_test, et_oof_train)
res_et = load_results_from_json('model_et.json')
print("ET-CV: {}".format(mean_absolute_error(y_train, res_et['oof_train'])))


if False:
    rf_params = {
        'n_jobs': 4,
        'n_estimators': 10,
        'max_features': 0.2,
        'max_depth': 8,
        'min_samples_leaf': 2,
        'verbose': 1,
    }
    
    rf = SklearnWrapper(clf=RandomForestRegressor, seed=SEED, params=rf_params)
    rf_oof_train, rf_oof_test = get_oof(rf)
    save_results_to_json('model_rf', rf_params, rf_oof_test, rf_oof_train)
# res_rf = load_results_from_json('model_rf.json')
# print("RF-CV: {}".format(mean_absolute_error(y_train, rf_oof_train)))


xgb_params = {
    'seed': 0,
    'colsample_bytree': 0.7,
    'silent': 0,
    'subsample': 0.7,
    'learning_rate': 0.075,
    'objective': 'reg:linear',
    'max_depth': 2,
    'num_parallel_tree': 1,
    'min_child_weight': 1,
    'eval_metric': 'mae',
    'nrounds': 30,
    'verbose': 1,
}

#  xg = XgbWrapper(seed=SEED, params=xgb_params)
#  xg_oof_train, xg_oof_test = get_oof(xg)
#  print("XG-CV: {}".format(mean_absolute_error(y_train, xg_oof_train)))
#


lightgbm_params_fair = {
    'exec_path': '../../../LightGBM/lightgbm',
    'config': '',
    'application': 'regression-fair',
    'num_iterations': 20000,
    'learning_rate': 0.002,
    'num_leaves': 31,
    'tree_learner': 'serial',
    'num_threads': 4,
    'min_data_in_leaf': 100,
    'metric': 'l1',
    'feature_fraction': 0.9,
    'feature_fraction_seed': SEED,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'bagging_seed': SEED,
    'metric_freq': 500,
    'early_stopping_round': 100,
}

# lg_fair = LightgbmWrapper(seed=SEED, params=lightgbm_params_fair)
# lg_oof_train_fair, lg_oof_test_fair = get_oof(lg_fair)
res_lg_fair = load_results_from_json('model_fair_c_2_w_100_lr_0.002_trees_20K.json')
print("LG_Fair-CV: {}".format(mean_absolute_error(y_train, res_lg_fair['oof_train'])))


lightgbm_params_l2 = {
    'exec_path': '../../../LightGBM/lightgbm',
    'config': '',
    'application': 'regression',
    'num_iterations': 7000,
    'learning_rate': 0.1,
    'num_leaves': 31,
    'tree_learner': 'serial',
    'num_threads': 4,
    'min_data_in_leaf': 100,
    'metric': 'l1exp',
    'feature_fraction': 0.9,
    'feature_fraction_seed': SEED,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'bagging_seed': SEED,
    'metric_freq': 1,
    'early_stopping_round': 100,
    'verbose': True
}

lg_l2 = LightgbmWrapper(seed=SEED, params=lightgbm_params_l2)
lg_oof_train_l2, lg_oof_test_l2 = get_oof(lg_l2)
# res_lg_l2 = load_results_from_json('model_l2_lr_0.01_trees_7K.json')
print("LG_L2-CV: {}".format(mean_absolute_error(y_train, res_lg_l2['oof_train'])))



res_array = [res_lg_fair, res_lg_l2, res_et]

for i, r in enumerate(res_array):
    cv_err  = np.abs(y_train - r['oof_train'].flatten())
    cv_mean = np.mean(cv_err)
    cv_std  = np.std(cv_err)
    print ("Model {0}: \tName = {1}, \tCV = {2}+{3}".format(i, r['name'], cv_mean, cv_std))

x_train = np.concatenate([r['oof_train'] for r in res_array], axis=1)
x_test  = np.concatenate([r['oof_test']  for r in res_array], axis=1)

print("{},{}".format(x_train.shape, x_test.shape))

dtrain = xgb.DMatrix(x_train, label=y_train)
dtest = xgb.DMatrix(x_test)

xgb_params = {
    'seed': 0,
    'colsample_bytree': 0.8,
    'silent': 1,
    'subsample': 0.6,
    'learning_rate': 0.01,
    'objective': 'reg:linear',
    'max_depth': 4,
    'num_parallel_tree': 1,
    'min_child_weight': 1,
    'eval_metric': 'mae',
}

res = xgb.cv(xgb_params, dtrain, num_boost_round=500, nfold=NFOLDS, seed=SEED,
             stratified=False, early_stopping_rounds=25, verbose_eval=10, show_stdv=True)

best_nrounds = res.shape[0] - 1
cv_mean = res.iloc[-1, 0]
cv_std = res.iloc[-1, 1]

print('Ensemble-CV: {0}+{1}'.format(cv_mean, cv_std))

gbdt = xgb.train(xgb_params, dtrain, best_nrounds)

submission = pd.read_csv(SUBMISSION_FILE)
submission.iloc[:, 1] = gbdt.predict(dtest)
submission.to_csv('xgstacker_starter.sub.csv', index=None)

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
from sklearn.grid_search import GridSearchCV
from bayes_opt import BayesianOptimization
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

    def train(self, x_train, y_train, x_valid, y_valid):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)


class XgbWrapper(object):
    def __init__(self, seed=0, params=None):
        self.param = params
        self.param['seed'] = seed
        self.nrounds = params.pop('nrounds', 10000)
        self.shift = params.pop('shift', 200)
        self.num_folds_trained = 1

    def evalerror(self, preds, dtrain):
        return 'mae', mean_absolute_error(np.exp(preds) - self.shift,
                                          np.exp(dtrain.get_label()) - self.shift)

    def train(self, x_train, y_train, x_valid, y_valid):
        dtrain = xgb.DMatrix(x_train, label=np.log(y_train + self.shift))
        eval_error_func = lambda p,d : self.evalerror(p,d)
        # if self.num_folds_trained == 0:
        #     res = xgb.cv(self.param, dtrain, num_boost_round=self.nrounds, nfold=5, seed=SEED,
        #                  stratified=False, early_stopping_rounds=25, verbose_eval=10, show_stdv=False,
        #                  feval=eval_error_func)
        #     best_nrounds = res.shape[0] - 1
        #     cv_mean = res.iloc[-1, 0]
        #     print ('[XgbWrapper.train.cv] CV mean = {:.4f}'.format(cv_mean))
        #     print ('[XgbWrapper.train.cv] Best Rounds = {}'.format(best_nrounds))
        #     self.nrounds = best_nrounds
        # self.num_folds_trained += 1

        # dvalid = xgb.DMatrix(x_valid, label=np.log(y_valid + self.shift))
        # self.gbdt = xgb.train(self.param, dtrain, self.nrounds,
        #                       evals=[(dvalid, 'valid')], verbose_eval=10, feval=eval_error_func,
        #                       early_stopping_rounds=25)

        self.gbdt = xgb.train(self.param, dtrain, self.nrounds)

    def predict(self, x):
        return np.exp(self.gbdt.predict(xgb.DMatrix(x))) - self.shift


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


def load_results_from_csv(oof_train_filename, oof_test_filename):
    res = dict({'name': oof_train_filename})
    res['oof_train'] = np.array(pd.read_csv(oof_train_filename)['loss']).reshape(-1,1)
    res['oof_test']  = np.array(pd.read_csv(oof_test_filename)['loss']).reshape(-1,1)
    return res


res_keras = load_results_from_csv('keras_1/oof_train_keras_400_0.4_200_0.2_nbags_4_nepochs_55_nfolds_5.csv', 'keras_1/submission_keras_400_0.4_200_0.2_nbags_4_nepochs_55_nfolds_5.csv')
print("Keras_1-CV: {}".format(mean_absolute_error(y_train, res_keras['oof_train'])))

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


if False:
    xgb_params = {
        'seed': 0,
        'colsample_bytree': 0.5,
        'silent': 1,
        'subsample': 0.8,
        'learning_rate': 0.01,
        'max_depth': 12,
        'num_parallel_tree': 1,
        'min_child_weight': 1,
        'nrounds': 1800,
        'alpha': 1,
        'gamma': 1,
        'verbose_eval': 10,
        'shift': 200
        # 'verbose': 1,
        # 'eval_metric': 'mae',
        # 'objective': 'reg:linear',
    }
    xg = XgbWrapper(seed=SEED, params=xgb_params)
    xgb_oof_train, xgb_oof_test = get_oof(xg)
    # save_results_to_json('model_xgb_1', xgb_params, xgb_oof_test, xgb_oof_train)
    # save_results_to_json('model_xgb_2', xgb_params, xgb_oof_test, xgb_oof_train)
    # save_results_to_json('model_xgb_3', xgb_params, xgb_oof_test, xgb_oof_train) # Fix incorrect sign of shift term in predictions from model_2
res_xgb = load_results_from_json('model_xgb_3.json')
print("XG-CV: {}".format(mean_absolute_error(y_train, res_xgb['oof_train'])))



if False:
    # lightgbm_params_fair = {
    #     'exec_path': '../../../LightGBM/lightgbm',
    #     'config': '',
    #     'application': 'regression-fair',
    #     'num_iterations': 20000,
    #     'learning_rate': 0.002,
    #     'num_leaves': 31,
    #     'tree_learner': 'serial',
    #     'num_threads': 4,
    #     'min_data_in_leaf': 100,
    #     'metric': 'l1',
    #     'feature_fraction': 0.9,
    #     'feature_fraction_seed': SEED,
    #     'bagging_fraction': 0.8,
    #     'bagging_freq': 5,
    #     'bagging_seed': SEED,
    #     'metric_freq': 500,
    #     'early_stopping_round': 100,
    # }
    # lg_fair = LightgbmWrapper(seed=SEED, params=lightgbm_params_fair)
    # lg_oof_train_fair, lg_oof_test_fair = get_oof(lg_fair)
    # save_results_to_json('model_fair_c_2_w_100_lr_0.002_trees_20K', lightgbm_params_fair, lg_oof_test_fair, lg_oof_train_fair)
    lightgbm_params_fair = {
        'exec_path': '../../../LightGBM/lightgbm',
        'config': '',
        'application': 'regression-fair',
        'fair_constant': 15.16,
        'fair_scaling': 194.388,
        'num_iterations': 50000,
        'learning_rate': 0.00588,
        'num_leaves': 107,
        'tree_learner': 'serial',
        'num_threads': 4,
        'min_data_in_leaf': 2,
        'metric': 'l1',
        'feature_fraction': 0.6665121,
        'feature_fraction_seed': SEED,
        'bagging_fraction': 0.96029,
        'bagging_freq': 3,
        'bagging_seed': SEED,
        'metric_freq': 100,
        'early_stopping_round': 100,
    }
    lg_fair = LightgbmWrapper(seed=SEED, params=lightgbm_params_fair)
    lg_oof_train_fair, lg_oof_test_fair = get_oof(lg_fair)
    save_results_to_json('model_fair_c_15.16_w_194.388_lr_0.00588_trees_50K', lightgbm_params_fair, lg_oof_test_fair, lg_oof_train_fair)
# res_lg_fair = load_results_from_json('model_fair_c_15.16_w_194.388_lr_0.00588_trees_50K.json')
res_lg_fair = load_results_from_json('model_fair_c_2_w_100_lr_0.002_trees_20K.json')
print("LG_Fair-CV: {}".format(mean_absolute_error(y_train, res_lg_fair['oof_train'])))


if False:
    # lightgbm_params_l2 = {
    #     'exec_path': '../../../LightGBM/lightgbm',
    #     'config': '',
    #     'application': 'regression',
    #     'num_iterations': 7000,
    #     'learning_rate': 0.1,
    #     'num_leaves': 31,
    #     'tree_learner': 'serial',
    #     'num_threads': 4,
    #     'min_data_in_leaf': 100,
    #     'metric': 'l1exp',
    #     'feature_fraction': 0.9,
    #     'feature_fraction_seed': SEED,
    #     'bagging_fraction': 0.8,
    #     'bagging_freq': 5,
    #     'bagging_seed': SEED,
    #     'metric_freq': 1,
    #     'early_stopping_round': 100,
    #     'verbose': True
    # }
    # lightgbm_params_l2 = {
    #     'exec_path': '../../../LightGBM/lightgbm',
    #     'config': '',
    #     'application': 'regression',
    #     'num_iterations': 20000,
    #     'learning_rate': 0.01745188,
    #     'num_leaves': 60,
    #     'tree_learner': 'serial',
    #     'num_threads': 4,
    #     'min_data_in_leaf': 482,
    #     'metric': 'l1exp',
    #     'feature_fraction': 0.2800091,
    #     # 'feature_fraction_seed': SEED,
    #     'bagging_fraction': 0.96443795,
    #     'bagging_freq': 3,
    #     # 'bagging_seed': SEED,
    #     'metric_freq': 1,
    #     'early_stopping_round': 100,
    #     'verbose': True
    # }
    lightgbm_params_l2 = {
        'exec_path': '../../../LightGBM/lightgbm',
        'config': '',
        'application': 'regression',
        'num_iterations': 2000,
        'learning_rate': 0.0251188,
        'num_leaves': 107,
        'tree_learner': 'serial',
        'num_threads': 4,
        'min_data_in_leaf': 215,
        'metric': 'l1exp',
        'feature_fraction': 0.6665121,
        # 'feature_fraction_seed': SEED,
        'bagging_fraction': 0.9602939,
        'bagging_freq': 3,
        # 'bagging_seed': SEED,
        'metric_freq': 10,
        'early_stopping_round': 100,
        'verbose': True
    }
    lg_l2 = LightgbmWrapper(seed=SEED, params=lightgbm_params_l2)
    lg_oof_train_l2, lg_oof_test_l2 = get_oof(lg_l2)
    # save_results_to_json('model_l2_bopt_run1_index75', lightgbm_params_l2, lg_oof_test_l2, lg_oof_train_l2)
    # save_results_to_json('model_l2_bopt_run2_index92', lightgbm_params_l2, lg_oof_test_l2, lg_oof_train_l2)
res_lg_l2_1 = load_results_from_json('model_l2_lr_0.01_trees_7K.json')
res_lg_l2_2 = load_results_from_json('model_l2_bopt_run1_index75.json')
res_lg_l2_3 = load_results_from_json('model_l2_bopt_run2_index92.json')
# print("LG_L2-CV: {}".format(mean_absolute_error(y_train, res_lg_l2['oof_train'])))
print("LG_L2 ({})-CV: {}".format(res_lg_l2_1['name'], mean_absolute_error(y_train, res_lg_l2_1['oof_train'])))
print("LG_L2 ({})-CV: {}".format(res_lg_l2_2['name'], mean_absolute_error(y_train, res_lg_l2_2['oof_train'])))
print("LG_L2 ({})-CV: {}".format(res_lg_l2_3['name'], mean_absolute_error(y_train, res_lg_l2_3['oof_train'])))

res_xgb_vlad_bs1_fast = load_results_from_json('../stacked_ensembles/test_xgb_A1-20161103T205502.json')
res_array = [res_lg_fair,
             # res_et,
             res_keras,
             res_lg_l2_1,
             res_lg_l2_2,
             res_lg_l2_3,
             res_xgb,
             res_xgb_vlad_bs1_fast,
            ]

for i, r in enumerate(res_array):
    cv_err  = np.abs(y_train - r['oof_train'].flatten())
    cv_mean = np.mean(cv_err)
    cv_std  = np.std(cv_err)
    print ("Model {0}: \tCV = {2:.3f}+{3:.1f}, \tName = {1} ".format(i, r['name'], cv_mean, cv_std))

x_train = np.concatenate([r['oof_train'] for r in res_array], axis=1)
x_test  = np.concatenate([r['oof_test' ] for r in res_array], axis=1)

print("{},{}".format(x_train.shape, x_test.shape))

dtrain = xgb.DMatrix(x_train, label=y_train)
dtest = xgb.DMatrix(x_test)

xgb_params = {
    'seed': 0,
    'colsample_bytree': 0.8,
    'silent': 1,
    'subsample': 0.6,
    'learning_rate': 0.003,
    'objective': 'reg:linear',
    'max_depth': 6,
    'num_parallel_tree': 1,
    'min_child_weight': 1,
    'eval_metric': 'mae',
}

res = xgb.cv(xgb_params, dtrain, num_boost_round=5000, nfold=NFOLDS, seed=SEED,
             stratified=False, early_stopping_rounds=25, verbose_eval=10, show_stdv=True)

best_nrounds = res.shape[0] - 1
cv_mean = res.iloc[-1, 0]
cv_std = res.iloc[-1, 1]

print('Ensemble-CV: {0:.3f}+{1:.1f}'.format(cv_mean, cv_std))
print('Best Rounds: {}'.format(best_nrounds))

gbdt = xgb.train(xgb_params, dtrain, int(best_nrounds * (1)))

submission = pd.read_csv(SUBMISSION_FILE)
submission.iloc[:, 1] = gbdt.predict(dtest)
submission.to_csv('xgstacker.sub.csv', index=None)

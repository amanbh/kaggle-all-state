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
from bayes_opt import BayesianOptimization
from IPython.display import display
import json
import pickle
import sys
from load_data import *
from level0_training import *

NFOLDS = 5
SEED = 0

(x_train, x_test, y_train, ntrain, ntest) = load_data(DATA_DIR="../../input")
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


def get_oof_score(clf):
    oof_train, oof_test = get_oof(clf)
    return mean_absolute_error(y_train, oof_train)


def get_oof_score_lgbm_ext(clf):
    import subprocess
    NFOLDS = 5
    oof_train = np.zeros((x_train.shape[0],1))
    num_predicted = 0
    for fold_id in range(NFOLDS):
        run_args = [clf.params['exec_path'],
                        "task=train",
                        "output_model=LightGBM_model.txt",
                        "min_sum_hessian_in_leaf=5.0",
                        "max_bin=255",
                        "boosting_type=gbdt",
                        "is_training_metric=False",
                        "data=./train_fold_{}.csv".format(fold_id),
                        "test_data=./cv_fold_{}.csv".format(fold_id),
                        "objective={}".format(clf.params['application']),
                        "learning_rate={}".format(clf.params['learning_rate']),
                        "num_leaves={}".format(clf.params['num_leaves']),
                        "tree_learner={}".format(clf.params['tree_learner']),
                        "num_threads={}".format(clf.params['num_threads']),
                        "min_data_in_leaf={}".format(clf.params['min_data_in_leaf']),
                        "metric={}".format(clf.params['metric']),
                        "feature_fraction={}".format(clf.params['feature_fraction']),
                        #"feature_fraction_seed={}".format(clf.params['feature_fraction_seed']),
                        "bagging_fraction={}".format(clf.params['bagging_fraction']),
                        "bagging_freq={}".format(clf.params['bagging_freq']),
                        #"bagging_seed={}".format(clf.params['bagging_seed']),
                        "metric_freq={}".format(clf.params['metric_freq']),
                        "early_stopping_round={}".format(clf.params['early_stopping_round']),
                        "num_iterations={}".format(clf.params['num_iterations']) ]
        # display(run_args)
        # print ('  '.join(run_args))
        subprocess.run(run_args)
        # print('-'*70)
        subprocess.run([clf.params['exec_path'],
                        "task=predict",
                        "input_model=LightGBM_model.txt",
                        "data=./cv_as_test_fold_{}.csv".format(fold_id) ])
        # print('-'*70)
        submission = pd.read_csv('LightGBM_predict_result.txt', header = None)
        oof_train[num_predicted:num_predicted+len(submission)] = np.exp(submission)
        num_predicted += len(submission)

    return mean_absolute_error(y_train, oof_train)
    


kf = KFold(ntrain, n_folds=NFOLDS, shuffle=False, random_state=SEED)

# Bayesian Opt for ExtraTreeRegressor hyper params
def et_bo_oof_score(n_estimators, max_features, max_depth, min_samples_leaf):
    et_params = {
        'n_jobs': 4,
        'n_estimators': int(n_estimators),
        'max_features': max_features,
        'max_depth': int(max_depth),
        'min_samples_leaf': int(min_samples_leaf),
        'verbose': 0,
    }
    et = SklearnWrapper(clf=ExtraTreesRegressor, seed=SEED, params=et_params)
    return -1.0 * get_oof_score(et)


if False:
    etBO = BayesianOptimization(et_bo_oof_score, {'n_estimators': (10, 200),
                                                  'max_features': (0.1, 0.999),
                                                  'max_depth':    (3, 12), 
                                                  'min_samples_leaf': (2, 25) })
    
    etBO.explore({'n_estimators': [2, 100],
                  'max_features': [0.9, 0.6],
                  'max_depth': [5, 12],
                  'min_samples_leaf': [2, 2]})
                                
    etBO.maximize()
                            

# Bayesian Opt for LightGBM hyper params
def lgbm_bo_oof_score(num_leaves, learning_rate, min_data_in_leaf, feature_fraction, bagging_fraction, bagging_freq):
    lightgbm_params_l2 = {
        'exec_path': '../../../LightGBM/lightgbm',
        'config': '',
        'application': 'regression',
        'num_iterations': 20000,
        'learning_rate': learning_rate,
        'num_leaves': int(num_leaves),
        'tree_learner': 'serial',
        'num_threads': 4,
        'min_data_in_leaf': int(min_data_in_leaf),
        'metric': 'l1exp',
        'feature_fraction': feature_fraction,
        'feature_fraction_seed': SEED,
        'bagging_fraction': bagging_fraction,
        'bagging_freq': int(bagging_freq),
        'bagging_seed': SEED,
        'metric_freq': 1000,
        'early_stopping_round': 5,
        'verbose': False,
        # 'min_sum_hessian_in_leaf': 5.0,
    }
    display(lightgbm_params_l2)
    # print("learning_rate={} num_leaves={} min_data_in_leaf={} feature_fraction={} bagging_fraction={} bagging_freq={}".format(learning_rate, int(num_leaves), int(min_data_in_leaf), feature_fraction, bagging_fraction, int(bagging_freq)))
    lg_l2 = LightgbmWrapper(seed=SEED, params=lightgbm_params_l2)
    return -1.0 * get_oof_score_lgbm_ext(lg_l2)


if True:
    print ("Bayesian-Optimization of LightGBM-L2 models")
    lgl2BO = BayesianOptimization(lgbm_bo_oof_score,
                                  {'num_leaves': (3, 127),
                                   'learning_rate': (0.003, 0.1),
                                   'min_data_in_leaf': (3,1000),
                                   'feature_fraction': (0.1, 0.999),
                                   'bagging_fraction': (0.1, 0.999),
                                   'bagging_freq':  (1,8) })
    #lgl2BO.initialize({-1140.09681: 
    lgl2BO.explore(
                            {'num_leaves': [31],
                             'learning_rate': [0.01],
                             'min_data_in_leaf': [100],
                             'feature_fraction': [0.9],
                             'bagging_fraction': [0.8],
                             'bagging_freq': [5] } 
                       #})
                       )

    iter_kappa = [(10,5), (30,3), (20,2), (20,1)]
    for i, (n_iter, kappa) in enumerate(iter_kappa):
        lgl2BO.maximize(n_iter = n_iter, kappa = kappa)
        print('-'*53)
        print('LGBM-L2: %f' % lgl2BO.res['max']['max_val'])

    print('Final Results')
    print('LGBM-L2: %f' % lgl2BO.res['max']['max_val'])



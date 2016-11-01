import pandas as pd
import numpy as np
from IPython.display import display
import pickle
import json
import sys

import xgboost as xgb
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from pylightgbm.models import GBMRegressor
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_absolute_error


#####################  Load Data  ###########################

def load_data(DATA_DIR="../../input", USE_PICKLED=True):
    """
    Load train.csv and test.csv from DATA_DIR
    Returns (x_train, x_test, y_train, ntrain, ntest)
    Also, saves a pickled file data.pkl for later use.
    """
    ID = 'id'
    TARGET = 'loss'
    NROWS = None

    TRAIN_FILE = "{0}/train.csv".format(DATA_DIR)
    TEST_FILE  = "{0}/test.csv".format(DATA_DIR)

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
        print ("Loading training/testing data with pickled data.pkl")
        with open('data.pkl', 'rb') as pkl_file:
            (x_train, x_test, y_train) = pickle.load(pkl_file)
            ntrain = x_train.shape[0]
            ntest  = x_test.shape[0]

    return (x_train, x_test, y_train, ntrain, ntest)


def load_submission(DATA_DIR="../../input"):
    """
    Returns sample_submission.csv as a pd.DataFrame
    """
    SUBMISSION_FILE = "{0}/sample_submission.csv".format(DATA_DIR)
    submission = pd.read_csv(SUBMISSION_FILE)
    return submission


def split_and_save_folds(NFOLDS=5, train_logloss=True, logloss_shift=1):
    (x_train, x_test, y_train, ntrain, ntest) = load_data()
    from sklearn.cross_validation import KFold
    kf = KFold(ntrain, n_folds=NFOLDS, shuffle=False, random_state=None)
    for i, (train_index, test_index) in enumerate(kf):
        x_tr = x_train[train_index]
        x_te = x_train[test_index]
        if train_logloss:
            y_tr = np.log1p(y_train[train_index] + logloss_shift - 1)
            y_te = np.log1p(y_train[test_index] + logloss_shift - 1)
        else:
            y_tr = y_train[train_index]
            y_te = y_train[test_index]
        tr_fold = pd.DataFrame(np.concatenate((y_tr.reshape(-1,1), x_tr), axis = 1))
        cv_fold = pd.DataFrame(np.concatenate((y_te.reshape(-1,1), x_te), axis = 1))
        cv_test_fold = pd.DataFrame(x_te)
        tr_fold.to_csv("train_fold_{}.csv".format(i), index = None, header = False)
        cv_fold.to_csv("cv_fold_{}.csv".format(i), index = None, header = False)
        cv_test_fold.to_csv("cv_as_test_fold_{}.csv".format(i), index = None, header = False)


def generate_logshift_processors(shift=200):
    preprocessor = lambda y: np.log(y + shift)
    postprocessor = lambda p: np.exp(p) - shift
    return preprocessor, postprocessor


################### CLF Wrappers ########################

class ClfWrapper(object):
    def __init__(self, params=None):
        self.preprocess_labels = params.pop('preprocess_labels', lambda x : x)
        self.postprocess_labels = params.pop('postprocess_labels', lambda x : x)


class SklearnWrapper(ClfWrapper):
    def __init__(self, clf, seed=0, params=None):
        ClfWrapper.__init__(self, params)
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train, x_valid, y_valid):
        self.clf.fit(x_train, self.preprocess_labels(y_train))

    def predict(self, x):
        return self.postprocess_labels(self.clf.predict(x))


class XgbWrapper(ClfWrapper):
    def __init__(self, seed=0, params=None):
        super().__init__(self, params)
        self.param = params
        self.param['seed'] = seed
        self.nrounds = params.pop('nrounds', 10000)
        self.num_folds_trained = 1

    def evalerror(self, preds, dtrain):
        return 'mae', mean_absolute_error(self.postprocess_labels(preds),
                                          self.postprocess_labels(dtrain.get_label()))

    def train(self, x_train, y_train, x_valid, y_valid):
        dtrain = xgb.DMatrix(x_train, label=self.preprocess_labels(y_train))
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
        return self.postprocess_labels(self.gbdt.predict(xgb.DMatrix(x)))


class LightgbmWrapper(ClfWrapper):
    def __init__(self, seed=0, params=None, pyLightGBM_managed=True):
        super().__init__(self, params)
        self.params = params
        if pyLightGBM_managed:
            self.clf = GBMRegressor(**params)
        self.params['pyLightGBM_managed'] = pyLightGBM_managed
        self.params['seed'] = seed

    def train(self, x_train, y_train, x_valid, y_valid):
        if self.params['pyLightGBM_managed']:
            self.clf.fit(x_train, self.preprocess_labels(y_train),
                         [(x_valid, self.preprocess_labels(y_valid))])
        else:
            print('Not ready for external LightGBM yet!')

    def predict(self, x):
        if self.params['pyLightGBM_managed']:
            return self.postprocess_labels(self.clf.predict(x))
        else:
            print('Not ready for external LightGBM yet!')


#################  CLF Training ############################


def get_fold(ntrain, NFOLDS=5, seed=0):
    kf = KFold(ntrain, n_folds=NFOLDS, shuffle=False, random_state=seed)
    return kf


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



###################  Level-0  Training   ##########################

def train_skl_model(model_key=None, clf=None, params=None):
    """
    Saves results dict for ExtraTreesRegressor
    """
    if params is None:
        if clf is ExtraTReesRegressor:
            params = {
                'n_jobs': 4,
                'n_estimators': 169,
                'max_features': 0.999,
                'max_depth': 12,
                'min_samples_leaf': 12,
                'verbose': 1,
            }
        elif clf is RandomForestRegressor:
            params = {
                'n_jobs': 4,
                'n_estimators': 10,
                'max_features': 0.2,
                'max_depth': 8,
                'min_samples_leaf': 2,
                'verbose': 1,
            }
        else:
            print('No known clf passed to train_skl_model and params are also missing!')
            return
    skl = SklearnWrapper(clf=clf, params=params)
    oof_train, oof_test = get_oof(skl)
    save_results_to_json(model_key, params, oof_test, oof_train)
    res_skl = load_results_from_json(model_key + '.json')
    print("{} CV: {}".format(
                        res_skl['name'],
                        mean_absolute_error(y_train, res_skl['oof_train']))
                        )


def train_xgb_model(model_key='model_xgb', xgb_params=None):
    """
    Saves results dict for XGB
    """
    if xgb_params is None:
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
    xg = XgbWrapper(params=xgb_params)
    xgb_oof_train, xgb_oof_test = get_oof(xg)
    # save_results_to_json('model_xgb_1', xgb_params, xgb_oof_test, xgb_oof_train)
    # save_results_to_json('model_xgb_2', xgb_params, xgb_oof_test, xgb_oof_train)
    # save_results_to_json('model_xgb_3', xgb_params, xgb_oof_test, xgb_oof_train) # Fix incorrect sign of shift term in predictions from model_2
    save_results_to_json(model_key, xgb_params, xgb_oof_test, xgb_oof_train)
    res_xgb = load_results_from_json(model_key + '.json')
    print("XG ({}) CV: {}".format(
                            res_xgb['name'],
                            mean_absolute_error(y_train, res_xgb['oof_train']))
                            )


def train_lgbm_model(model_key='model_lgbm', lgbm_params=None):
    if lgbm_params is None:
        lgbm_params = {
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
    lg = LightgbmWrapper(params=lgbm_params)
    oof_train, oof_test = get_oof(lg)
    save_results_to_json(model_key, lgbm_params, oof_test, oof_train)
    res_lg = load_results_from_json(model_key + '.json')
    print("LG {} CV: {}".format(
                            res_lg['name'],
                            mean_absolute_error(y_train, res_lg['oof_train']))
                            )


###################### Stacking ##################################

def load_model_results(model_key=None):
    """ Return results dict for model corresponding to model_key """
    res_dict = {'keras_1':
                    load_results_from_csv(
                      'keras_1/oof_train_keras_400_0.4_200_0.2_nbags_4_nepochs_55_nfolds_5.csv',
                      'keras_1/submission_keras_400_0.4_200_0.2_nbags_4_nepochs_55_nfolds_5.csv'
                    ),
                 'model_et':
                    load_results_from_json(
                      'model_et.json'
                    ),
                 'model_rf':
                    load_results_from_json(
                      'model_rf.json'
                    ),
                }[model_key]
    print("{} CV: {}".format(
                        res_dict['name'],
                        mean_absolute_error(y_train, res_dict['oof_train']))
                        )
    return res_dict



def load_all_model_results():
    model_keys = ['lg_fair',
                  'keras_1',
                  'model_et',
                  'model_rf',
                  'lg_l2_1',
                  'lg_l2_2',
                  'lg_l2_3',
                  'res_xgb']
    res_array = [ load_model_results(model_key) for model_key in model_keys ]


def train_level1_xgb_model(res_array):
    for i, r in enumerate(res_array):
        cv_err  = np.abs(y_train - r['oof_train'].flatten())
        cv_mean = np.mean(cv_err)
        cv_std  = np.std(cv_err)
        print ("Model {0}: \tCV = {2:.3f}+{3:.1f}, \tName = {1} ".format(i, r['name'], cv_mean, cv_std))

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
    print('Best Rounds: {}'.format(best_nrounds))

    gbdt = xgb.train(xgb_params, dtrain, int(best_nrounds * (1)))

    submission = pd.read_csv(SUBMISSION_FILE)
    submission.iloc[:, 1] = gbdt.predict(dtest)
    submission.to_csv('xgstacker_starter.sub.csv', index=None)


#################   Experiment ##########################

# if __name__ == "__main__":

(x_train, x_test, y_train, ntrain, ntest) = load_data()
NFOLDS = 5
kf = get_fold(ntrain, NFOLDS)
preprocessor, postprocessor = generate_logshift_processors(shift=200)
params = {
          'max_depth': 6,
          'max_features': 0.2,
          'min_samples_leaf': 2,
          'n_estimators': 4,
          'n_jobs': 4,
          'verbose': 1,
          'preprocess_labels': preprocessor,
          'postprocess_labels': postprocessor,
         }
train_skl_model(model_key='test_skl_rf', clf=RandomForestRegressor, params=params)

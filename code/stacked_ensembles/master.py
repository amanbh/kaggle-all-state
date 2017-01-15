import pandas as pd
import numpy as np
from IPython.display import display
import pickle
import json
import sys
import time

import xgboost as xgb
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from pylightgbm.models import GBMRegressor
from sklearn.cross_validation import KFold, StratifiedKFold
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.linear_model import SGDRegressor, HuberRegressor, Ridge
import itertools
from sklearn import preprocessing


#####################  Load Data  ###########################

### custom sorting - from Ali - https://www.kaggle.com/aliajouz/allstate-claims-severity/xgb-model/code
def custom_sorting(mylist):
    mylist_len=[]
    for i in mylist:
        mylist_len.append(len(str(i)))

    all_list=[]
    for i in np.unique(sorted(mylist_len)):
        i_list=[]
        for j in mylist:
            if len(j)==i:
                i_list.append(j)
        all_list=all_list + i_list
    return(all_list)

def load_data(DATA_DIR="../../input", USE_PICKLED=True, PREPROCESS_CONTS=False, PREPROCESS_CATS=False):
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
    if PREPROCESS_CONTS or PREPROCESS_CATS:
        USE_PICKLED = False

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
        if False:
            for feat in cats:
                print('Checking ', feat)
                utr = train[feat].unique()
                ute = test[feat].unique()
                remove_train = set(utr) - set(ute)
                remove_test = set(ute) - set(utr)
                remove = remove_train.union(remove_test)
                def filter_cat(x):
                    if x in remove:
                        return np.nan
                    return x

                if len(remove) > 0:
                    print('Pruning feature {} by mapping {}/{} values to nan'.format(feat, len(remove_train), len(remove_test)))
                    train_test[feat] = train_test[feat].apply(lambda x: filter_cat(x), 1)
        if PREPROCESS_CATS:
            # From Ali - https://www.kaggle.com/aliajouz/allstate-claims-severity/xgb-model/code 
            for cat in cats:
                mylist=(np.unique(train_test[cat])).tolist()
                sorting_list=custom_sorting(mylist)
                train_test[cat]=pd.Categorical(train_test[cat], sorting_list)
                train_test=train_test.sort_values(cat)
                train_test[cat] = pd.factorize(train_test[cat], sort=True)[0]

        # for feat in cats:
        #     train_test[feat] = pd.factorize(train_test[feat], sort=True)[0]

        if PREPROCESS_CONTS:
            train_test["cont1"] = (preprocessing.minmax_scale(train_test["cont1"]))**(1/4)
            train_test["cont2"] = (preprocessing.minmax_scale(train_test["cont2"]))**(1/4)    
            train_test["cont3"] = (preprocessing.minmax_scale(train_test["cont3"]))**(4)
            train_test["cont4"] = np.sqrt(preprocessing.minmax_scale(train_test["cont4"]))
            train_test["cont5"] = (preprocessing.minmax_scale(train_test["cont5"]))**2
            train_test["cont6"] = np.exp(preprocessing.minmax_scale(train_test["cont6"]))
            train_test["cont7"] = (preprocessing.minmax_scale(train_test["cont7"]))**4
            train_test["cont8"] = (preprocessing.minmax_scale(train_test["cont8"]))**(1/4)
            train_test["cont9"] = (preprocessing.minmax_scale(train_test["cont9"]))**4
            train_test["cont10"] = np.log1p(preprocessing.minmax_scale(train_test["cont10"]))
            train_test["cont11"] = (preprocessing.minmax_scale(train_test["cont11"]))**4
            train_test["cont12"] = (preprocessing.minmax_scale(train_test["cont12"]))**4
            train_test["cont13"] = np.log(preprocessing.minmax_scale(train_test["cont13"]))
            train_test["cont14"] = (preprocessing.minmax_scale(train_test["cont14"]))**4

        if False:
            train_test["cont1"] = np.sqrt(MinMaxScaler().fit_transform(train_test["cont1"]))
            train_test["cont4"] = np.sqrt(MinMaxScaler().fit_transform(train_test["cont4"]))
            train_test["cont5"] = np.sqrt(MinMaxScaler().fit_transform(train_test["cont5"]))
            train_test["cont8"] = np.sqrt(MinMaxScaler().fit_transform(train_test["cont8"]))
            train_test["cont10"] = np.sqrt(MinMaxScaler().fit_transform(train_test["cont10"]))
            train_test["cont11"] = np.sqrt(MinMaxScaler().fit_transform(train_test["cont11"]))
            train_test["cont12"] = np.sqrt(MinMaxScaler().fit_transform(train_test["cont12"]))
            train_test["cont6"] = np.log(MinMaxScaler().fit_transform(train_test["cont6"])+0000.1)
            train_test["cont7"] = np.log(MinMaxScaler().fit_transform(train_test["cont7"])+0000.1)
            train_test["cont9"] = np.log(MinMaxScaler().fit_transform(train_test["cont9"])+0000.1)
            train_test["cont13"] = np.log(MinMaxScaler().fit_transform(train_test["cont13"])+0000.1)
            train_test["cont14"]=(np.maximum(train_test["cont14"]-0.179722,0)/0.665122)**0.25

        print ("Head ( train_test ) : ")
        print (train_test.head())

        x_train = np.array(train_test.iloc[:ntrain,:])
        x_test = np.array(train_test.iloc[ntrain:,:])
        if not PREPROCESS_CONTS:
            with open('data.pkl', 'wb') as pkl_file:
                pickle.dump( (x_train, x_test, y_train), pkl_file)
    else:
        print ("Loading training/testing data with pickled data.pkl")
        with open('data.pkl', 'rb') as pkl_file:
            (x_train, x_test, y_train) = pickle.load(pkl_file)
            ntrain = x_train.shape[0]
            ntest  = x_test.shape[0]

    return (x_train, x_test, y_train, ntrain, ntest)


def load_data_skew_cat_comb_std_scale():
    ID = 'id'
    TARGET = 'loss'
    # with open('../kernel-lexical-encoding-feat-comb-xgb-1109/data.skew.feat_comb.ss.pkl', 'rb') as pkl_file:
    # with open('../kernel-lexical-encoding-feat-comb-xgb-1109/data.skew.remove_cat.feat_comb.ss.pkl', 'rb') as pkl_file:
    # with open('../kernel-lexical-encoding-feat-comb-xgb-1109/data.skew.sort_cat.feat_comb.ali_cont.pkl', 'rb') as pkl_file:
    with open('../kernel-lexical-encoding-feat-comb-xgb-1109/data.sort_cat.feat_comb.ali_cont.pkl', 'rb') as pkl_file:
    
        (Xy_train, X_test) = pickle.load(pkl_file)
        y_train = np.exp(Xy_train[TARGET].ravel()) - 200
        x_train = Xy_train.drop([ID, TARGET], axis=1).as_matrix()
        x_test = X_test.drop([ID], axis=1).as_matrix()
        ntrain = x_train.shape[0]
        ntest  = x_test.shape[0]

    return (x_train, x_test, y_train, ntrain, ntest)


def add_linearized_conts(ADD_DIFF=True, ADD_DIFF_OLD=False):
    global x_train
    global x_test
    lin_train = pd.read_csv('../../input/all-the-allstate-dates-eda/lin_train.csv')
    lin_test = pd.read_csv('../../input/all-the-allstate-dates-eda/lin_test.csv')
    lin_feats = [feat for feat in lin_train.columns if 'lin_cont' in feat]
    for feat in lin_feats:
        print("Adding feature: {}".format(feat))
        print(x_train.shape)
        c = lin_train[feat].values.reshape(lin_train[feat].shape[0], 1)
        print(c.shape)
        x_train = np.concatenate((x_train, c), axis=1)
        c = lin_test[feat].values.reshape(lin_test[feat].shape[0], 1)
        x_test = np.concatenate((x_test, c), axis=1)
    if ADD_DIFF:
        for comb in itertools.combinations(lin_feats, 2):
            print("Adding sum and diff of {} and {}".format(comb[0], comb[1]))
            comb_diff = lin_train[comb[0]] - lin_train[comb[1]]
            comb_sum = lin_train[comb[0]] + lin_train[comb[1]]
            comb_diff = comb_diff.values.reshape(comb_diff.shape[0], 1)
            comb_sum = comb_sum.values.reshape(comb_sum.shape[0], 1)
            x_train = np.concatenate((x_train, comb_diff), axis=1)
            x_train = np.concatenate((x_train, comb_sum), axis=1)
            comb_diff = lin_test[comb[0]] - lin_test[comb[1]]
            comb_sum = lin_test[comb[0]] + lin_test[comb[1]]
            comb_diff = comb_diff.values.reshape(comb_diff.shape[0], 1)
            comb_sum = comb_sum.values.reshape(comb_sum.shape[0], 1)
            x_test = np.concatenate((x_test, comb_diff), axis=1)
            x_test = np.concatenate((x_test, comb_sum), axis=1)
    elif ADD_DIFF_OLD:
        print("Adding diff of lin_con1 and lin_cont9")
        diff = lin_train['lin_cont1'] - lin_train['lin_cont9']
        diff = diff.values.reshape(diff.shape[0], 1)
        x_train = np.concatenate((x_train, diff), axis=1)
        diff = lin_test['lin_cont1'] - lin_test['lin_cont9']
        diff = diff.values.reshape(diff.shape[0], 1)
        x_test = np.concatenate((x_test, diff), axis=1)


def add_poly_features(x_train=None, x_test=None, num_features=None):
    if num_features is None:
        poly = PolynomialFeatures(degree=2)
        x_train = poly.fit_transform(x_train)
        x_test  = poly.fit_transform(x_test)
        return (x_train, x_test)
    else:
        cols = np.random.choice(range(x_train.shape[1]), replace=True, size=(num_features,2))
        new_train = np.zeros((x_train.shape[0],x_train.shape[1]+num_features))
        new_test  = np.zeros((x_test.shape[0],x_test.shape[1]+num_features))
        new_train[:,:-num_features] = x_train
        new_test[:,:-num_features] = x_test
        for i, c in enumerate(cols):
            # print("Adding column {} as product of columns {} and {}".format(i, c[0], c[1]))
            new_train[:,x_train.shape[1]+i] = np.product(x_train[:,c], axis=1)
            new_test[:,x_test.shape[1]+i] = np.product(x_test[:,c], axis=1)
        return (new_train, new_test)


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


def get_timestamp():
    return time.strftime("%Y%m%dT%H%M%S")


def generate_logshift_processors(shift=200):
    preprocessor = lambda y: np.log(y + shift)
    postprocessor = lambda p: np.exp(p) - shift
    name = 'log_shift_' + repr(shift)
    return preprocessor, postprocessor, name


def generate_powershift_processors(degree=0.25, shift=1):
    preprocessor = lambda y: (y + shift) ** degree
    postprocessor = lambda p: (p ** (1/degree)) - shift
    name = 'pow_' + repr(degree) + '_shift_' + repr(shift)
    return preprocessor, postprocessor, name


def logregobj(preds, dtrain):
    labels = dtrain.get_label()
    con = 0.7
    x = (preds-labels) # * np.log(labels**4 + 9000)
    grad = con*x / (np.abs(x)+con)
    hess = con**2 / (np.abs(x)+con)**2
    return grad, hess


def loss_func_cauchy(preds, dtrain):
    labels = dtrain.get_label()
    con = 1
    x = preds-labels
    grad = x / (1+(x/con)**2)
    hess = 1 / (1+(x/con)**2)
    return grad, hess


################### CLF Wrappers ########################

class ClfWrapper(object):
    def __init__(self, params=None):
        self.param = params.copy()
        self.preprocess_labels = self.param.pop('preprocess_labels', lambda x : x)
        self.postprocess_labels = self.param.pop('postprocess_labels', lambda x : x)
        self.label_processor_function_name = self.param.pop('label_processor_function_name', 'identity')

    def get_oof(self):
        print("x_train.shape = ", x_train.shape)
        print("x_test.shape  = ", x_test.shape)
        oof_train = np.zeros((ntrain,))
        oof_test = np.zeros((ntest,))
        oof_test_skf = np.empty((NFOLDS, ntest))

        for i, (train_index, test_index) in enumerate(kf):
            print ('Start Training Fold {}'.format(i))
            x_tr = x_train[train_index]
            y_tr = y_train[train_index]
            x_te = x_train[test_index]
            y_te = y_train[test_index]

            self.train(x_tr, y_tr, x_te, y_te)
            oof_train[test_index] = self.predict(x_te)
            oof_test_skf[i, :] = self.predict(x_test)

        oof_test[:] = oof_test_skf.mean(axis=0)
        return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)


    def get_oof_score(self):
        oof_train, oof_test = get_oof(self)
        return mean_absolute_error(y_train, oof_train)


class SklearnWrapper(ClfWrapper):
    def __init__(self, clf, seed=0, params=None):
        ClfWrapper.__init__(self, params)
        if clf != HuberRegressor:
            self.param['random_state'] = seed
        self.clf = clf(**self.param)

    def train(self, x_train, y_train, x_valid, y_valid):
        print("Starting training")
        self.clf.fit(x_train, self.preprocess_labels(y_train))

    def predict(self, x):
        return self.postprocess_labels(self.clf.predict(x))


class XgbWrapper(ClfWrapper):
    def __init__(self, seed=0, params=None):
        ClfWrapper.__init__(self, params)
        self.param['seed'] = seed
        self.nrounds = self.param.pop('nrounds', 10000)
        # self.num_folds_trained = 1
        self.verbose_eval = self.param.pop('verbose_eval', False)
        self.early_stopping_rounds = self.param.pop('early_stopping_rounds', 100)
        self.objective = self.param.pop('obj', None)

    def evalerror(self, preds, dtrain):
        return 'MAE', mean_absolute_error(self.postprocess_labels(preds),
                                          self.postprocess_labels(dtrain.get_label()))

    def train(self, x_train, y_train, x_valid, y_valid):
        dtrain = xgb.DMatrix(x_train, label=self.preprocess_labels(y_train))
        dvalid = xgb.DMatrix(x_valid, label=self.preprocess_labels(y_valid))
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
        print(self.param)
        self.gbdt = xgb.train(self.param, dtrain, self.nrounds,
                              obj=logregobj, #loss_func_cauchy, # logregobj, # self.objective,
                              early_stopping_rounds=self.early_stopping_rounds,
                              evals=[(dtrain, 'Training'), (dvalid, 'Validation')], feval=eval_error_func,
                              verbose_eval=self.verbose_eval
                              )

    def predict(self, x):
        return self.postprocess_labels(self.gbdt.predict(xgb.DMatrix(x), ntree_limit=self.gbdt.best_ntree_limit))


class LightgbmWrapper(ClfWrapper):
    def __init__(self, seed=0, params=None, pyLightGBM_managed=True):
        ClfWrapper.__init__(self, params)
        if pyLightGBM_managed:
            self.clf = GBMRegressor(**self.param)
        self.param['pyLightGBM_managed'] = pyLightGBM_managed
        self.param['seed'] = seed

    def train(self, x_train, y_train, x_valid, y_valid):
        if self.param['pyLightGBM_managed']:
            self.clf.fit(x_train, self.preprocess_labels(y_train),
                         [(x_valid, self.preprocess_labels(y_valid))])
        else:
            print('Not ready for external LightGBM yet!')

    def predict(self, x):
        if self.param['pyLightGBM_managed']:
            return self.postprocess_labels(self.clf.predict(x))
        else:
            print('Not ready for external LightGBM yet!')


#################  CLF Training ############################


def get_fold(ntrain, NFOLDS=5, seed=0):
    kf = KFold(ntrain, n_folds=NFOLDS, shuffle=False, random_state=seed)
    return kf


def get_stratified_fold(y_train, NFOLDS=10, seed=0):
    perc_values = np.percentile(y_train, range(10,101,10))
    y_perc = np.zeros_like(y_train)
    for v in perc_values[:-1]:
        y_perc += (y_train > v)
    folds = StratifiedKFold(y_perc, n_folds=NFOLDS, shuffle=True, random_state=None)
    return folds


def save_results_to_json(name, params, oof_test, oof_train):
    # Remove functions which cannot be saved as JSON
    local_params = params.copy()
    keys = list(local_params.keys())
    for k in keys:
        if callable(local_params[k]):
            local_params.pop(k)
    with open(name + '.json', 'w') as outfile:
        json.dump({ 'name': name,
                    'params': local_params,
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
        if clf is ExtraTreesRegressor:
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
    oof_train, oof_test = skl.get_oof()
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
        print('No params passed to train_xgb_model')
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
    xgb_oof_train, xgb_oof_test = xg.get_oof()
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
            'application': 'regression',
            # 'fair_constant': 15.16,
            # 'fair_scaling': 194.388,
            'num_iterations': 2400,
            'learning_rate': 0.01,
            'num_leaves': 200,
            'tree_learner': 'serial',
            'num_threads': 2,
            'min_data_in_leaf': 8,
            'metric': 'l1exp',
            'feature_fraction': 0.3,
            'feature_fraction_seed': SEED,
            'bagging_fraction': 0.8,
            'bagging_freq': 100,
            'bagging_seed': SEED,
            'metric_freq': 50,
            'early_stopping_round': 100,
        }
    lg = LightgbmWrapper(params=lgbm_params, pyLightGBM_managed=True)
    oof_train, oof_test = lg.get_oof()
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
                    lambda : load_results_from_csv(
                                'keras_1/oof_train_keras_400_0.4_200_0.2_nbags_4_nepochs_55_nfolds_5.csv',
                                'keras_1/submission_keras_400_0.4_200_0.2_nbags_4_nepochs_55_nfolds_5.csv'
                    ),
                 'model_et':
                    lambda: load_results_from_json(
                      'model_et.json'
                    ),
                 'model_rf':
                    lambda : load_results_from_json(
                      'model_rf.json'
                    ),
                }
    res_func = res_dict.get(model_key, lambda : load_results_from_json(model_key+'.json'))
    res = res_func()
    print("{} CV: {}".format(
                        res['name'],
                        mean_absolute_error(y_train, res['oof_train']))
                        )
    return res



def load_all_model_results(model_keys=None):
    """
    Returns a list of result dicts
    by calling load_model_results on all given model_keys.

    Parameters
    ----------
    model_key : list
        A list of strings corresponding to the model_key

    Returns
    -------
    res_array : list of dicts
        A list of dict objects.
        Each object contains results correspoding to model_key.
    """
    if model_keys is None:
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
(x_train, x_test, y_train, ntrain, ntest) = load_data_skew_cat_comb_std_scale()  # load_data(PREPROCESS_CONTS=True, PREPROCESS_CATS=False)  # load_data_skew_cat_comb_std_scale() #load_data() # load_data(PREPROCESS_CONTS=True)
# num_quad_features = 1000 # int(sys.argv[1])
# (x_train, x_test) = add_poly_features(x_train, x_test, num_features = num_quad_features)
add_linearized_conts(ADD_DIFF=False, ADD_DIFF_OLD=True)

NFOLDS = 10
print("NFOLDS = {}".format(NFOLDS))
# kf = get_fold(ntrain, NFOLDS)
kf = get_stratified_fold(y_train, NFOLDS)
# preprocessor, postprocessor, prep_name = generate_logshift_processors(shift=200)
preprocessor, postprocessor, prep_name = generate_powershift_processors(degree=0.32, shift=1)
# params = {
#           # 'max_depth': 12,
#           # 'max_features': 0.999,
#           # 'min_samples_leaf': 12,
#           # 'n_estimators': 400,
#           # 'n_jobs': 4,
#           # 'verbose': 1,
#           'alpha': 1e-3,
#           # 'epsilon': 2,
#           # 'max_iter': 4000,
#           'normalize': True,
#           'preprocess_labels': preprocessor,
#           'postprocess_labels': postprocessor,
#           'label_processor_function_name': prep_name,
#          }
# timestamp =  '20161208T000000' #get_timestamp() # 
# # model_key = 'skl_Ridge-' + 'quad-'+ repr(num_quad_features) + timestamp
# model_key = 'skl_Ridge-' + 'lin-comb_quad-' + timestamp
# train_skl_model(model_key=model_key,
#                 clf=Ridge, # RandomForestRegressor,
#                 params=params)
# res = load_model_results(model_key=model_key)
# sub = load_submission()
# sub['loss'] = res['oof_test']
# sub.to_csv('submission-' + timestamp + '.csv', index=False)

# xgb_params = {
#     # 'alpha': 1,
#     'base_score': 1,
#     'colsample_bytree': 0.7,
#     # 'gamma': 0.5290,
#     'learning_rate': 0.03,
#     'max_depth': 12,
#     'min_child_weight': 100,
#     'num_parallel_tree': 1,
#     'seed': 12468,
#     'silent': 1,
#     'subsample': 0.7,
#     # 'verbose': 1,
#     # 'eval_metric': 'mae',
#     # 'objective': 'reg:linear',
#     # 'objective': lambda p, d: logregobj(p, d),
#     'nrounds': 640,
#     'verbose_eval': 10,
#     'early_stopping_rounds': 100,
#     'preprocess_labels': preprocessor,
#     'postprocess_labels': postprocessor,
#     'label_processor_function_name': prep_name,
# }

mean_base_score = np.mean(preprocessor(y_train)) + 1.5
print("mean for base_score = ", mean_base_score)
xgb_params = {
    # 'alpha': 1,
    # 'gamma': 1,
    'base_score': mean_base_score,
    'colsample_bytree': 0.08,
    'learning_rate': 0.01,
    'max_depth': 12,
    'min_child_weight': 100,
    'seed': 35791,
    'silent': 1,
    'subsample': 0.7,
    'booster': 'gbtree',
    # 'verbose': 1,
    # 'eval_metric': 'mae',
    # 'objective': 'reg:linear',
    # 'objective': lambda p, d: logregobj(p, d),
    'nrounds': 4000,
    'verbose_eval': 10,
    'early_stopping_rounds': 100,
    'preprocess_labels': preprocessor,
    'postprocess_labels': postprocessor,
    'label_processor_function_name': prep_name,
}
timestamp = get_timestamp()
if True:
    model_key = 'xgb-mrooijer-lin-attempt_13-obj_fair0.7-preprocessor_' + prep_name + '-base_score-' + repr(mean_base_score) + '-' + timestamp
    train_xgb_model(model_key=model_key, xgb_params=xgb_params)
else:
    # model_key = 'test_xgb_A1-20161103T205502'
    del model_key

res_xgb = load_model_results(model_key=model_key)
sub = load_submission()
sub['loss'] = res_xgb['oof_test']
sub.to_csv('submission-' + timestamp + '.csv', index=False)

# SEED = 500
# lgbm_params = {
#         'exec_path': '../../../LightGBM/lightgbm',
#         'config': '',
#         'application': 'regression-fair',
#         'num_iterations': 3500,
#         'learning_rate': 0.01,
#         'num_leaves': 2000,
#         'tree_learner': 'serial',
#         'num_threads': 4,
#         'min_data_in_leaf': 100,
#         'metric': 'l1',
#         'feature_fraction': 0.08,
#         'feature_fraction_seed': SEED,
#         'bagging_fraction': 0.7,
#         'bagging_freq': 10,
#         'bagging_seed': SEED,
#         'metric_freq': 200,
#         'early_stopping_round': 100,
#         # 'preprocess_labels': preprocessor,
#         # 'postprocess_labels': postprocessor,
#         # 'label_processor_function_name': prep_name,
#         'min_sum_hessian_in_leaf': 10.0,
#         'fair_const': 1.5,
#         'fair_scaling': 100.0,
#         }
# timestamp = get_timestamp()
# model_key = "lgbm_fair_c_1.5_w_100-leaf_1000-lr_0.003-max_40K_trees-" + timestamp
# train_lgbm_model(model_key=model_key, lgbm_params=lgbm_params)
# res_lgbm = load_model_results(model_key=model_key)
# sub = load_submission()
# sub['loss'] = res_lgbm['oof_test']
# sub.to_csv('submission-' + timestamp + '.csv', index=False)

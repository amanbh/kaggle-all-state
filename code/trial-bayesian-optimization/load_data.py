# -*- coding: utf-8 -*-
"""
Load train/test data
"""

import pandas as pd
import numpy as np
from IPython.display import display
import pickle

def load_data(DATA_DIR="../../input"):
    """
    Load train.csv and test.csv from DATA_DIR
    Returns (x_train, x_test, y_train, ntrain, ntest)
    ALso, saves a pickled file data.pkl for later use.
    """
    ID = 'id'
    TARGET = 'loss'
    NROWS = None

    TRAIN_FILE = "{0}/train.csv".format(DATA_DIR)
    TEST_FILE  = "{0}/test.csv".format(DATA_DIR)
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

if __name__ == "__main__":
    split_and_save_folds(NFOLDS = 5)


'''
Author: Danijel Kivaranovic
Title: Neural network (Keras) with sparse data
'''
import time
# time.sleep(6000)

## import libraries
import numpy as np
np.random.seed(123)

import pandas as pd
import subprocess
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import KFold, StratifiedKFold
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Lambda
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint, ProgbarLogger
from keras.regularizers import l1

## Batch generators ##################################################################################################################################

def batch_generator(X, y, batch_size, shuffle):
    #chenglong code for fiting from generator (https://www.kaggle.com/c/talkingdata-mobile-user-demographics/forums/t/22567/neural-network-for-sparse-matrices)
    number_of_batches = np.ceil(X.shape[0]/batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    if shuffle:
        np.random.shuffle(sample_index)
    while True:
        batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
        X_batch = X[batch_index,:].toarray()
        y_batch = y[batch_index]
        counter += 1
        yield X_batch, y_batch
        if (counter == number_of_batches):
            if shuffle:
                np.random.shuffle(sample_index)
            counter = 0

def batch_generatorp(X, batch_size, shuffle):
    number_of_batches = X.shape[0] / np.ceil(X.shape[0]/batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    while True:
        batch_index = sample_index[batch_size * counter:batch_size * (counter + 1)]
        X_batch = X[batch_index, :].toarray()
        counter += 1
        yield X_batch
        if (counter == number_of_batches):
            counter = 0

########################################################################################################################################################

## read data
train = pd.read_csv('../../input/train.csv')
test = pd.read_csv('../../input/test.csv')

ADD_LINEARIZED_CONTS = True
ADD_DIFF = False
if ADD_LINEARIZED_CONTS:
    lin_train = pd.read_csv('../../input/all-the-allstate-dates-eda/lin_train.csv')
    lin_test = pd.read_csv('../../input/all-the-allstate-dates-eda/lin_test.csv')
    lin_feats = [feat for feat in lin_train.columns if 'lin_cont' in feat]
    for feat in lin_feats:
        print("Adding feature: {}".format(feat))
        train[feat] = lin_train[feat]
        test[feat] = lin_test[feat]
    if ADD_DIFF:
        print("Adding diff of lin_con1 and lin_cont9")
        train['diff_lin_cont1_lin_cont9'] = lin_train['lin_cont1'] - lin_train['lin_cont9']
        test['diff_lin_cont1_lin_cont9'] = lin_test['lin_cont1'] - lin_test['lin_cont9']
    del (lin_train, lin_test, lin_feats)

## set test loss to NaN
test['loss'] = np.nan

## response and IDs
y = train['loss'].values
id_train = train['id'].values
id_test = test['id'].values

## stack train test
ntrain = train.shape[0]
tr_te = pd.concat((train, test), axis = 0)

## Preprocessing and transforming to sparse data
sparse_data = []

f_cat = [f for f in tr_te.columns if 'cat' in f]
for f in f_cat:
    dummy = pd.get_dummies(tr_te[f].astype('category'))
    tmp = csr_matrix(dummy)
    sparse_data.append(tmp)

f_num = [f for f in tr_te.columns if 'cont' in f]
scaler = StandardScaler()
tmp = csr_matrix(scaler.fit_transform(tr_te[f_num]))
sparse_data.append(tmp)

del(tr_te, train, test)

## sparse train and test data
xtr_te = hstack(sparse_data, format = 'csr')
xtrain = xtr_te[:ntrain, :]
xtest = xtr_te[ntrain:, :]

print('Dim train', xtrain.shape)
print('Dim test', xtest.shape)

del(xtr_te, sparse_data, tmp)

## neural net
def nn_model():
    model = Sequential()
    model.add(Dense(400, input_dim = xtrain.shape[1], init = 'he_normal'))
    # model.add(Dense(400, input_dim = xtrain.shape[1], init = 'he_normal', W_regularizer=l1(0.01)))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(Dropout(0.4))
    model.add(Dense(200, init = 'he_normal'))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(Dropout(0.2))
    model.add(Dense(50, init = 'he_normal'))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(Dropout(0.2))
    model.add(Dense(1, init = 'he_normal'))
    # model.add(Lambda(lambda x: x + 7.79))
    # model.add(Lambda(lambda x: np.exp(x)))
    # model.add(Lambda(lambda x: x-200))
    model.compile(loss = 'mae', optimizer = 'adadelta')
    return(model)


## train models
nfolds = 5
nbags = 1
nepochs = 100
pred_oob = np.zeros(xtrain.shape[0])
pred_test = np.zeros(xtest.shape[0])

# folds = KFold(len(y), n_folds = nfolds, shuffle = False, random_state = None)
perc_values = np.percentile(y, range(10,101,10))
y_perc = np.zeros_like(y)
for v in perc_values[:-1]:
   y_perc += (y > v)
folds = StratifiedKFold(y_perc, n_folds=nfolds, shuffle=True, random_state=None)

for j in range(nbags):
    for i, (inTr, inTe) in enumerate(folds):
        xtr = xtrain[inTr]
        ytr = (y[inTr] + 1) ** 0.32
        xte = xtrain[inTe]
        yte = (y[inTe] + 1) ** 0.32
        pred = np.zeros(xte.shape[0])
        model = nn_model()
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto'),
            ModelCheckpoint('keras.best.hdf5', monitor='val_loss', save_best_only=True, save_weights_only=False, verbose=1),
            ]
        fit = model.fit_generator(generator = batch_generator(xtr, ytr, 128, True),
                                  nb_epoch = nepochs,
                                  samples_per_epoch = xtr.shape[0],
                                  verbose = 2,
                                  callbacks = callbacks,
                                  validation_data = batch_generator(xte, yte, 800, False),
                                  nb_val_samples = xte.shape[0],
                                 )
        model = load_model('keras.best.hdf5')
        p = (model.predict_generator(generator = batch_generatorp(xte, 800, False), val_samples = xte.shape[0])[:,0] ** (1/0.32)) - 1
        print('Bag ', j, ' Fold ', i, ' - MAE:', mean_absolute_error(y[inTe], p))
        pred_oob[inTe] += p
        pred_test += (model.predict_generator(generator = batch_generatorp(xtest, 800, False), val_samples = xtest.shape[0])[:,0] ** (1/0.32)) - 1
    score = mean_absolute_error(y, pred_oob/(j+1))
    print('Bag ', j, '- MAE:', score)
pred_oob /= nbags
print('Total - MAE:', mean_absolute_error(y, pred_oob))

## train predictions
df = pd.DataFrame({'id': id_train, 'loss': pred_oob})
df.to_csv('oof_train.csv', index = False)

## test predictions
pred_test /= (nfolds*nbags)
df = pd.DataFrame({'id': id_test, 'loss': pred_test})
df.to_csv('oof_test.csv', index = False)


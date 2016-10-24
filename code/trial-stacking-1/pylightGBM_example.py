
# coding: utf-8

# # pyLigthGBM
# 
# Python wrapper for Microsoft [LightGBM](https://github.com/Microsoft/LightGBM)
# 
# **GitHub      :  [https://github.com/ArdalanM/pyLightGBM](https://github.com/ArdalanM/pyLightGBM) **
# 
# ---
# 

# In[4]:

import os, gc
import numpy as np
import pandas as pd

from sklearn import datasets, metrics, model_selection
from sklearn.preprocessing import LabelEncoder

from pylightgbm.models import GBMRegressor


# ### DATA
# 
# For this example used data from Kaggle competition _Allstate Claims Severity_  
# You can download data from from Kaggle website : https://www.kaggle.com/c/allstate-claims-severity/data

# In[5]:

df_train = pd.read_csv("../../all-state/datasets/train.csv.zip")
print('Train data shape', df_train.shape)

df_test = pd.read_csv("../../all-state/datasets/test.csv.zip")
print('Test data shape', df_test.shape)


# Extracting `loss` from train and `id` from test

# In[6]:

y = np.log(df_train['loss']+1).as_matrix().astype(np.float)
id_test = np.array(df_test['id'])


# Merging train and test data

# In[7]:

df = df_train.append(df_test, ignore_index=True)
del df_test, df_train
gc.collect()

print('Merged data shape', df.shape)


# Droping not useful columns

# In[8]:

df.drop(labels=['loss', 'id'], axis=1, inplace=True)


# Transfrom categorical features `cat` from 1 to 116

# In[9]:

le = LabelEncoder()

for col in df.columns.tolist():
    if 'cat' in col:
        df[col] = le.fit_transform(df[col])


# ### TRAIN, VALIDATION, TEST
# Split data into train, validation(for early stopping) and test set

# In[10]:

print('train-test split')
df_train, df_test = df.iloc[:len(y)], df.iloc[len(y):]
del df
gc.collect()

print('train-validation split\n')
X = df_train.as_matrix()
X_train, X_valid, y_train, y_valid = model_selection.train_test_split(X, y, test_size=0.2, random_state=42)
X_test = df_test.as_matrix()

del df_train, df_test
gc.collect()

print('Train shape', X_train.shape)
print('Validation shape', X_valid.shape)
print('Test shape', X_test.shape)


# ### TRAINING GBMRegressor
# List of parameters and their explanation you can find here https://github.com/Microsoft/LightGBM/wiki/Quick-Start
# 
# **don't forget to change `exec_path` here**

# In[11]:

seed = 42

gbmr = GBMRegressor(
    exec_path='../../LightGBM/lightgbm', # change this to your LighGBM path
    config='',
    application='regression-fair',
    num_iterations=10000,
    learning_rate=0.002,
    num_leaves=31,
    tree_learner='serial',
    num_threads=1,
    min_data_in_leaf=100,
    metric='l1',
    feature_fraction=0.9,
    feature_fraction_seed=seed,
    bagging_fraction=0.8,
    bagging_freq=5,
    bagging_seed=seed,
    metric_freq=500,
    early_stopping_round=50
)

gbmr.fit(X_train, y_train, test_data=[(X_valid, y_valid)])
print("Mean Square Error: ", metrics.mean_absolute_error(y_true=(np.exp(y_valid)-1), y_pred=(np.exp(gbmr.predict(X_valid))-1)))


# Predicting Test set

# In[12]:

y_test_preds = gbmr.predict(X_test)
y_test_preds=(np.exp(y_test_preds)-1)


# Make submission file

# In[14]:

df_submision = pd.read_csv('../../all-state/datasets/sample_submission.csv.zip')
df_submision['loss'] = y_test_preds


# Save submission file

# In[15]:

df_submision.to_csv('submission.csv',index=False)


# #### This submision file scored 1138.06444 on LB


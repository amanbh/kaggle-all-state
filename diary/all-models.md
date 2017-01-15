# model_fair_c_2_w_100_lr_0.002_trees_20K
CV = 1145.92
LB = 1126.17
LightGBM with Fair objective on loss: 
    fair-constant = 2 and fair-scaling = 100 (LightGBM could not deal with small values of Hessian)
"params": {
    "bagging_fraction": 0.8,
    "bagging_seed": 0,
    "num_iterations": 20000,
    "num_leaves": 31,
    "application": "regression-fair",
    "min_data_in_leaf": 100,
    "learning_rate": 0.002,
    "num_threads": 4,
    "metric": "l1",
    "early_stopping_round": 100,
    "exec_path": "../../../LightGBM/lightgbm",
    "bagging_freq": 5,
    "metric_freq": 500,
    "seed": 0,
    "config": "",
    "feature_fraction_seed": 0,
    "feature_fraction": 0.9,
    "tree_learner": "serial"
},
NFOLDS = 5

# model_l2_lr_0.01_trees_7K
CV = 1138.99
LB = 1118.89
LightGBM with L2 on log-loss: 
"params": {
    "bagging_fraction": 0.8,
    "bagging_seed": 0,
    "num_iterations": 7000,
    "num_leaves": 31,
    "application": "regression",
    "min_data_in_leaf": 100,
    "learning_rate": 0.01,
    "num_threads": 4,
    "metric": "l1exp",
    "early_stopping_round": 100,
    "exec_path": "../../../LightGBM/lightgbm",
    "bagging_freq": 5,
    "metric_freq": 500,
    "seed": 0,
    "config": "",
    "feature_fraction_seed": 0,
    "feature_fraction": 0.9,
    "tree_learner": "serial"
}
NFOLDS = 5


----

# Fifth Stacking (with XGB)
LB = 1114.27
CV = 1131.16

Model 0:        CV = 1145.929+1581.4,   Name = model_fair_c_2_w_100_lr_0.002_trees_20K
Model 1:        CV = 1138.184+1546.8,   Name = keras_1/oof_train_keras_400_0.4_200_0.2_nbags_4_nepochs_55_nfolds_5.csv
Model 2:        CV = 1138.992+1560.6,   Name = model_l2_lr_0.01_trees_7K
Model 3:        CV = 1136.702+1569.7,   Name = model_l2_bopt_run1_index75
Model 4:        CV = 1136.977+1566.1,   Name = model_l2_bopt_run2_index92
Model 5:        CV = 1194.327+1463.7,   Name = model_xgb_2 (Incorrect sign of shift in postprocess_labels)

Level 1: XGB (default stacking)
Ensemble-CV: 1131.163+6.8 with 5 folds and 238 rounds
Final Submission with 238 rounds

----

# test_xgb_A1-20161103T205502.json
LB = 1115.1
CV = 1146.40
Vladimir XGB-1114 params + base_score=1 + higher eta
xgb_params = {
 'alpha': 1,
 'base_score': 1,
 'colsample_bytree': 0.5,
 'early_stopping_rounds': 50,
 'gamma': 1,
 'label_processor_function_name': 'log_shift_200',
 'learning_rate': 0.1,
 'max_depth': 12,
 'min_child_weight': 1,
 'nrounds': 500,
 'num_parallel_tree': 1,
 'seed': 0,
 'silent': 1,
 'subsample': 0.8,
 'verbose_eval': 10
 }

----

# Sixth Stacking (with XGB)
LB = 1113.49
CV = 1130.97
Model 0:        CV = 1145.929+1581.4,   Name = model_fair_c_2_w_100_lr_0.002_trees_20K
Model 1:        CV = 1138.184+1546.8,   Name = keras_1/oof_train_keras_400_0.4_200_0.2_nbags_4_nepochs_55_nfolds_5.csv
Model 2:        CV = 1138.992+1560.6,   Name = model_l2_lr_0.01_trees_7K
Model 3:        CV = 1136.702+1569.7,   Name = model_l2_bopt_run1_index75
Model 4:        CV = 1136.977+1566.1,   Name = model_l2_bopt_run2_index92
Model 5:        CV = 1135.422+1549.2,   Name = model_xgb_3
Model 6:        CV = 1146.402+1565.6,   Name = test_xgb_A1-20161103T205502)

Level 1: XGB 
(default stacking)                      Ensemble-CV: 1131.226+7.1 with 5 folds and 239 rounds
(default + max_depth increased to 6)    Ensemble-CV: 1131.008+6.8 with 5 folds and 238 rounds
(default + max_depth = 6, log(200+f))   Ensemble-CV: 1131.027+6.8 with 5 folds and 238 rounds
(default + max_depth = 6, eta = 0.003)  Ensemble-CV: 1130.972+6.8 with 5 folds and 803 rounds **This version is submitted**
Final Submission with 803 rounds

# xgb_Vladimir_base_score_1-20161103T230621
LB = 1113.16
CV = 1133.08
Vladimir XGB-1114 params + base_score=1
NFOLDS = 10
        {'alpha': 1,
         'base_score': 1,
         'colsample_bytree': 0.5,
         'early_stopping_rounds': 100,
         'gamma': 1,
         'label_processor_function_name': 'log_shift_200',
         'learning_rate': 0.01,
         'max_depth': 12,
         'min_child_weight': 1,
         'nrounds': 2300,
         'num_parallel_tree': 1,
         'seed': 0,
         'silent': 1,
         'subsample': 0.8,
         'verbose_eval': 10
        }


# Seventh Stacking (with XGB)
LB = 1112.94
CV = 1130.13
Model 0:        CV = 1145.929+1581.4,   Name = model_fair_c_2_w_100_lr_0.002_trees_20K
Model 1:        CV = 1138.184+1546.8,   Name = keras_1/oof_train_keras_400_0.4_200_0.2_nbags_4_nepochs_55_nfolds_5.csv
Model 2:        CV = 1138.992+1560.6,   Name = model_l2_lr_0.01_trees_7K
Model 3:        CV = 1136.702+1569.7,   Name = model_l2_bopt_run1_index75
Model 4:        CV = 1136.977+1566.1,   Name = model_l2_bopt_run2_index92
Model 5:        CV = 1135.422+1549.2,   Name = model_xgb_3
Model 6:        CV = 1146.402+1565.6,   Name = test_xgb_A1-20161103T205502
Model 7:        CV = 1133.082+1546.8,   Name = xgb_Vladimir_base_score_1-20161103T230621

Level 1 XGB (default + max_depth = 6, eta = 0.003)
    Ensemble-CV: 1130.132+7.0 with 5 folds and 801 rounds **This version is submitted**


# keras_2
Batch normalization; 3 hidden layers; EarlyStopping (patience 5)
CV = 1133.94
LB = 1114.67

    model = Sequential()
    model.add(Dense(400, input_dim = xtrain.shape[1], init = 'he_normal'))
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
    model.compile(loss = 'mae', optimizer = 'adadelta')
    return(model)
NFOLDS = 5
nbags = 5

# Eight Stacking (with XGB)
CV = 1128.42
LB = 1111.25
Model 0:        CV = 1145.929+1581.4,   Name = model_fair_c_2_w_100_lr_0.002_trees_20K
Model 1:        CV = 1138.184+1546.8,   Name = keras_1/oof_train_keras_400_0.4_200_0.2_nbags_4_nepochs_55_nfolds_5.csv
Model 2:        CV = 1133.942+1554.8,   Name = keras_2/oof_train.csv    **KERAS with Batch Norm**
Model 3:        CV = 1138.992+1560.6,   Name = model_l2_lr_0.01_trees_7K
Model 4:        CV = 1136.702+1569.7,   Name = model_l2_bopt_run1_index75
Model 5:        CV = 1136.977+1566.1,   Name = model_l2_bopt_run2_index92
Model 6:        CV = 1135.422+1549.2,   Name = model_xgb_3
Model 7:        CV = 1146.402+1565.6,   Name = test_xgb_A1-20161103T205502
Model 8:        CV = 1133.082+1546.8,   Name = xgb_Vladimir_base_score_1-20161103T230621

Level 1 XGB
(default + max_depth = 6, eta = 0.01, gamma = 1)                        Ensemble-CV: 1128.420+7.2 with 5 folds and 239 rounds
(default + max_depth = 6, eta = 0.01, gamma = 1, colsample=0.99)        Ensemble-CV: 1128.432+7.2 with 5 folds and 239 rounds
(default + max_depth = 6, eta = 0.003, gamma = 1)                       Ensemble-CV: 1128.222+7.1 with 5 folds and 804 rounds **This version is submitted**

# keras_3
Batch Normalization **after non-linear layer**; 3 hidden layers; Early Stopping (patience 5)
CV = 1135.62
LB = 1116.48
    model = Sequential()
    model.add(Dense(400, input_dim = xtrain.shape[1], init = 'he_normal'))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Dense(200, init = 'he_normal'))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(50, init = 'he_normal'))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(1, init = 'he_normal'))
    model.compile(loss = 'mae', optimizer = 'adadelta')
    return(model)


# Ninth Stacking (with XGB)
CV = 1127.88
LB = 1110.74

Model 0:        CV = 1145.929+1581.4,   Name = model_fair_c_2_w_100_lr_0.002_trees_20K
Model 1:        CV = 1138.184+1546.8,   Name = keras_1/oof_train_keras_400_0.4_200_0.2_nbags_4_nepochs_55_nfolds_5.csv
Model 2:        CV = 1133.942+1554.8,   Name = keras_2/oof_train.csv
Model 3:        CV = 1135.619+1564.4,   Name = **keras_3/oof_train.csv - Keras with BN layer moved to after non-linear layer**
Model 4:        CV = 1138.992+1560.6,   Name = model_l2_lr_0.01_trees_7K
Model 5:        CV = 1136.702+1569.7,   Name = model_l2_bopt_run1_index75
Model 6:        CV = 1136.977+1566.1,   Name = model_l2_bopt_run2_index92
Model 7:        CV = 1135.422+1549.2,   Name = model_xgb_3
Model 8:        CV = 1146.402+1565.6,   Name = test_xgb_A1-20161103T205502
Model 9:        CV = 1133.082+1546.8,   Name = xgb_Vladimir_base_score_1-20161103T230621

Level 1 XGB
{'colsample_bytree': 0.8,
 'eval_metric': 'mae',
 'gamma': 1,
 'learning_rate': 0.01,
 'max_depth': 6,
 'min_child_weight': 1,
 'num_parallel_tree': 1,
 'objective': 'reg:linear',
 'seed': 0,
 'subsample': 0.6}
(default + max_depth = 6, eta = 0.01, gamma = 1)                        Ensemble-CV: 1127.957+7.0 with 5 folds and 238 rounds
(default + max_depth = 6, eta = 0.003, gamma = 1)                       Ensemble-CV: 1127.884+7.0 with 5 folds and 803 rounds **This version is submitted**


# xgb-1117-attempt1-20161107T234751
CV = 1142.35
LB = 1115.81

NFOLDS = 5

{'alpha': 1,
 'base_score': 1,
 'colsample_bytree': 0.3085,
 'early_stopping_rounds': 25,
 'gamma': 0.529,
 'label_processor_function_name': 'log_shift_200',
 'learning_rate': 0.1,
 'max_depth': 7,
 'min_child_weight': 4.2922,
 'nrounds': 2300,
 'num_parallel_tree': 1,
 'seed': 0,
 'subsample': 0.993,
 'verbose_eval': 10}

**(above params)**                      CV = 1142.35
(-alpha,-base_score)                    CV = 1144.97
(eta=0.3)                               CV = 1159.31
(preprocess_conts)                      CV = 1142.92
(org preprocessing and script)          CV = 1145.32

# Tenth Stacking (with XGB)
CV = 1127.76
LB = 1110.62
Model 0:        CV = 1145.929+1581.4,   Name = model_fair_c_2_w_100_lr_0.002_trees_20K
Model 1:        CV = 1138.184+1546.8,   Name = keras_1/oof_train_keras_400_0.4_200_0.2_nbags_4_nepochs_55_nfolds_5.csv
Model 2:        CV = 1133.942+1554.8,   Name = keras_2/oof_train.csv
Model 3:        CV = 1135.619+1564.4,   Name = keras_3/oof_train.csv
Model 4:        CV = 1138.992+1560.6,   Name = model_l2_lr_0.01_trees_7K
Model 5:        CV = 1136.702+1569.7,   Name = model_l2_bopt_run1_index75
Model 6:        CV = 1136.977+1566.1,   Name = model_l2_bopt_run2_index92
Model 7:        CV = 1135.422+1549.2,   Name = model_xgb_3
Model 8:        CV = 1146.402+1565.6,   Name = test_xgb_A1-20161103T205502
Model 9:        CV = 1133.082+1546.8,   Name = xgb_Vladimir_base_score_1-20161103T230621
Model 10:       CV = 1142.349+1552.5,   Name = **xgb-1117-attempt1-20161107T234751**

Level 1 XGB
{'colsample_bytree': 0.8,
 'eval_metric': 'mae',
 'gamma': 1,
 'learning_rate': 0.003,
 'max_depth': 6,
 'min_child_weight': 1,
 'num_parallel_tree': 1,
 'objective': 'reg:linear',
 'seed': 0,
 'subsample': 0.6}
(default + max_depth = 6, eta = 0.003, gamma = 1)                       Ensemble-CV: 1127.755+6.9 with 5 folds and 806 rounds **This version is submitted**


# Stacking with model_et    **not-submitted**
Add model_et to Tenth Stacking
Gives CV = 1127.711+7.2 after 805 rounds


# keras_A4
CV = 1140.48
not submitted to LB

keras_3 + **batch size = 32**
    model = Sequential()
    model.add(Dense(400, input_dim = xtrain.shape[1], init = 'he_normal'))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Dense(200, init = 'he_normal'))
                                                                                                                                                                                                                                                                           model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(50, init = 'he_normal'))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(1, init = 'he_normal'))
    model.compile(loss = 'mae', optimizer = 'adadelta')


# Stacking with keras_A4 & skl_rf-20161108T012715 **not submitted**
CV = 1127.70
Model 0:        CV = 1145.929+1581.4,   Name = model_fair_c_2_w_100_lr_0.002_trees_20K
Model 1:        CV = 1269.171+1569.8,   Name = model_et
Model 2:        CV = 1223.941+1693.0,   Name = skl_rf-20161108T012715
Model 3:        CV = 1138.184+1546.8,   Name = keras_1/oof_train_keras_400_0.4_200_0.2_nbags_4_nepochs_55_nfolds_5.csv
Model 4:        CV = 1133.942+1554.8,   Name = keras_2/oof_train.csv
Model 5:        CV = 1135.619+1564.4,   Name = keras_3/oof_train.csv
Model 6:        CV = 1140.481+1563.9,   Name = keras_A4/oof_train.csv
Model 7:        CV = 1138.992+1560.6,   Name = model_l2_lr_0.01_trees_7K
Model 8:        CV = 1136.702+1569.7,   Name = model_l2_bopt_run1_index75
Model 9:        CV = 1136.977+1566.1,   Name = model_l2_bopt_run2_index92
Model 10:       CV = 1135.422+1549.2,   Name = model_xgb_3
Model 11:       CV = 1146.402+1565.6,   Name = test_xgb_A1-20161103T205502 
Model 12:       CV = 1133.082+1546.8,   Name = xgb_Vladimir_base_score_1-20161103T230621
Model 13:       CV = 1142.349+1552.5,   Name = xgb-1117-attempt1-20161107T234751

Level 1 (XGB)
{'colsample_bytree': 0.8,
 'eval_metric': 'mae',
 'gamma': 1,
 'learning_rate': 0.003,
 'max_depth': 6,
 'min_child_weight': 1,
 'num_parallel_tree': 1,
 'objective': 'reg:linear',
 'seed': 0,
 'silent': 1,
 'subsample': 0.6}
(default + max_depth = 6, eta = 0.003, gamma = 1)
(above params)                                    Ensemble-CV: 1127.700+7.1 Best Rounds: 807
(above params, NFOLDS = 10)                       Ensemble-CV: 1127.563+9.1 Best Rounds: 810


# AM_1 
**Mean of keras_2 and xgb_Vladimir_base_score_1-20161103T230621**
CV = 1124.74
LB = 1106.82

xgb_Vladimir_base_score_1-20161103T230621
CV = 1133.08
LB = 1113.16

keras_2
CV = 1133.94
LB = 1114.67


Arithematic Mean:   CV = 1124.736 **submitted**
Geometric Mean:     CV = 1125.02
Harmonic Mean:      CV = 1749.954


# keras_A5_1
CV = 1138.632
LB = 1119.769
**keras_2 + adam instead of adadelta**
def nn_model():
    model = Sequential()
    model.add(Dense(400, input_dim = xtrain.shape[1], init = 'he_normal'))
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
    model.compile(loss = 'mae', optimizer = 'adam')
    return(model)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto'),
]

nfolds = 5
nbags = 5
batch size = 128
Avg. epochs ~ 17 but patience should have been lower to avoid overfitting.
Time ~ 3h5m (down form 8h due to fewer avg epochs)


# keras_A5_2
CV = 1136.687
LB = 1119.54
**keras_A5_1 + EarlyStoppingPatience = 2**
def nn_model():
    model = Sequential()
    model.add(Dense(400, input_dim = xtrain.shape[1], init = 'he_normal'))
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
    model.compile(loss = 'mae', optimizer = 'adam')
    return(model)

callbacks = [
    EarlyStopping(monitor='val_loss', **patience=2**, verbose=0, mode='auto'),
]
nfolds = 5
nbags = 5
batch size = 128
Time ~ 2h25m (24% faster convergence than keras_A5_1)

# lgbm_l2_loglossshift_extremin_1112-20161110T023325
CV = 1135.93
LB = 1115.50

        lgbm_params = {
        'exec_path': '../../../LightGBM/lightgbm',
        'config': '',
        'application': 'regression',
        'num_iterations': 2400,
        'learning_rate': 0.01,
        'num_leaves': 200,
        'tree_learner': 'serial',
        'num_threads': 1,
        'min_data_in_leaf': 8,
        'metric': 'l1exp',
        'feature_fraction': 0.3,
        'feature_fraction_seed': SEED,
        'bagging_fraction': 0.8,
        'bagging_freq': 100,
        'bagging_seed': SEED,
        'metric_freq': 50,
        'early_stopping_round': 100,
        'preprocess_labels': preprocessor,
        'postprocess_labels': postprocessor,
        'label_processor_function_name': log_shift_200,
        }
NFOLDS = 5


# lgbm_l2_loglossshift_extremin_1112-lr_0.003-20161110T031110
CV = 1132.89
LB = 1113.76
lgbm_params = {
    'exec_path': '../../../LightGBM/lightgbm',
    'config': '',
    'application': 'regression',
    'num_iterations': **24000**,
    'learning_rate': **0.003**,
    'num_leaves': 200,
    'tree_learner': 'serial',
    'num_threads': 1,
    'min_data_in_leaf': 8,
    'metric': 'l1exp',
    'feature_fraction': 0.3,
    'feature_fraction_seed': SEED,
    'bagging_fraction': 0.8,
    'bagging_freq': 100,
    'bagging_seed': SEED,
    'metric_freq': 50,
    'early_stopping_round': **300**,
    'preprocess_labels': preprocessor,
    'postprocess_labels': postprocessor,
    'label_processor_function_name': log_shift_200,
    }
**NFOLDS = 10**


# keras_A5_11

CV = 1136.11
LB = 1118.65

**keras_A5_1 with NFOLDS = 10**
def nn_model():
    model = Sequential()
    model.add(Dense(400, input_dim = xtrain.shape[1], init = 'he_normal'))
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
    model.compile(loss = 'mae', optimizer = 'adam')
    return(model)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto'),
]

nfolds = 10
nbags = 5
batch size = 128
Avg. epochs ~ 17
Time ~ 7h


# keras_A6_1

**Add x -> x^2 neuron as last hidden layer**
NFOLDS = 5
...


# keras_A6_2

**Add x -> x^3 neuron as last hidden layer**
Lower epochs to 5, Increased nbags to 40
NFOLDS = 5
patience = 5
adam


# keras_A7_1

CV = 1138.80

**keras_2 + Increased dropout**
0.5, 0.4, 0.3
nbags = 5
patience = 5
epochs = 200


# keras_A8_1
**2 hidden layer NN**
CV = 1140.44
def nn_model():
    model = Sequential()
    model.add(Dense(200, input_dim = xtrain.shape[1], init = 'he_normal'))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(Dropout(0.3))
    model.add(Dense(100, init = 'he_normal'))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(Dropout(0.2))
    model.add(Dense(1, init = 'he_normal'))
    model.compile(loss = 'mae', optimizer = 'adam')
    return(model)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto'),
]

nfolds = 5
nbags = 5
epochs = 200
batch size = 256
Avg epochs ~ 22
Time ~ 1h40m

------------------------------------------------------------------------------------------------------------------------------------

# Checkpoint:

24 models trained.

Model 0:        CV = 1145.929+1581.4,   Name = model_fair_c_2_w_100_lr_0.002_trees_20K
Model 1:        CV = 1269.171+1569.8,   Name = model_et
Model 2:        CV = 1223.941+1693.0,   Name = skl_rf-20161108T012715
Model 3:        CV = 1138.184+1546.8,   Name = keras_1/oof_train_keras_400_0.4_200_0.2_nbags_4_nepochs_55_nfolds_5.csv
Model 4:        CV = 1133.942+1554.8,   Name = keras_2/oof_train.csv
Model 5:        CV = 1135.619+1564.4,   Name = keras_3/oof_train.csv
Model 6:        CV = 1140.481+1563.9,   Name = keras_A4/oof_train.csv
Model 7:        CV = 1138.992+1560.6,   Name = model_l2_lr_0.01_trees_7K
Model 8:        CV = 1136.702+1569.7,   Name = model_l2_bopt_run1_index75
Model 9:        CV = 1136.977+1566.1,   Name = model_l2_bopt_run2_index92
Model 10:       CV = 1135.422+1549.2,   Name = model_xgb_3
Model 11:       CV = 1146.402+1565.6,   Name = test_xgb_A1-20161103T205502
Model 12:       CV = 1133.082+1546.8,   Name = xgb_Vladimir_base_score_1-20161103T230621
Model 13:       CV = 1142.349+1552.5,   Name = xgb-1117-attempt1-20161107T234751
Model 14:       CV = 1124.736+1538.2,   Name = mean_keras_2_xgb_Vladimir_base_score_1-20161103T230621
Model 15:       CV = 1138.632+1564.7,   Name = ./keras_A5_1/oof_train.csv
Model 16:       CV = 1136.688+1567.8,   Name = ./keras_A5_2/oof_train.csv
Model 17:       CV = 1136.114+1558.9,   Name = ./keras_A5_11/oof_train.csv
Model 18:       CV = 1135.930+1563.4,   Name = lgbm_l2_loglossshift_extremin_1112-20161110T023325
Model 19:       CV = 1132.891+1557.3,   Name = lgbm_l2_loglossshift_extremin_1112-lr_0.003-20161110T031110
Model 20:       CV = 1146.014+1587.9,   Name = ./keras_A6_1/oof_train.csv
Model 21:       CV = 1146.831+1598.9,   Name = ./keras_A6_2/oof_train.csv
Model 22:       CV = 1138.803+1570.2,   Name = ./keras_A7_1/oof_train.csv
Model 23:       CV = 1140.436+1550.8,   Name = ./keras_A8_1/oof_train.csv

Top 5 Models (in terms of CV):
Model 14:       CV = 1124.736+1538.2,   Name = mean_keras_2_xgb_Vladimir_base_score_1-20161103T230621
Model 19:       CV = 1132.891+1557.3,   Name = lgbm_l2_loglossshift_extremin_1112-lr_0.003-20161110T031110      LB = 1113.76
Model 12:       CV = 1133.082+1546.8,   Name = xgb_Vladimir_base_score_1-20161103T230621                        LB = 1113.16
Model 4:        CV = 1133.942+1554.8,   Name = keras_2/oof_train.csv                                            LB = 1114.67
Model 10:       CV = 1135.422+1549.2,   Name = model_xgb_3
Model 5:        CV = 1135.619+1564.4,   Name = keras_3/oof_train.csv
Model 18:       CV = 1135.930+1563.4,   Name = lgbm_l2_loglossshift_extremin_1112-20161110T023325



------------------------------------------------------------------------------------------------------------------------------------

# First keras stacking with 24 models
LB = 1105.01
CV = 1125.09

nbags = 2
nfolds = 10
nepochs = 20
patience = 5


# Second keras stacking with 24 models
LB = 1105.03
CV = 1123.32

nbags = 72
nfolds = 10
nepochs = 20
patience = 5

# Weighted Avg:
LB = 1107.04
CV = 1123.947


Weights learned by :
def mae_loss_func(weights):
    ''' scipy minimize will pass the weights as a numpy array '''
    final_prediction = np.sum(weights * l1_x_train, axis=1)
    return mean_absolute_error(y_train, final_prediction)


starting_values = np.random.uniform(size=l1_x_train.shape[1])

cons = ({'type':'eq','fun':lambda w: 1-sum(w)})
bounds = [(0,1)]*l1_x_train.shape[1]

res = minimize(mae_loss_func, 
           starting_values, 
           method = 'SLSQP', 
           bounds = bounds, 
           constraints = cons,
           options={'maxiter': 100000})

best_score = res['fun']
weights = res['x']

Top Weights:
[(w, res_array[i]['name']) for i, w in  enumerate(weights) if w &gt; 0.025]
Out[13]: 
[(0.029924441852829425, 'model_fair_c_2_w_100_lr_0.002_trees_20K'),
 (0.23979125612889587, 'keras_2/oof_train.csv'),
 (0.03121021642706116, 'keras_3/oof_train.csv'),
 (0.026771362179366247, 'keras_A4/oof_train.csv'),
 (0.32766704844898437, 'xgb_Vladimir_base_score_1-20161103T230621'),
 (0.17725741042161941, './keras_A5_11/oof_train.csv'),
 (0.10315213368586121,
  'lgbm_l2_loglossshift500_extremin_1112-lr_0.01-20161113T203540'),
 (0.031877919601295351, './keras_A8_1/oof_train.csv')]

Models used:
Model 0:        CV = 1145.929+1581.4,   Name = model_fair_c_2_w_100_lr_0.002_trees_20K
Model 1:        CV = 1269.171+1569.8,   Name = model_et
Model 2:        CV = 1223.941+1693.0,   Name = skl_rf-20161108T012715
Model 3:        CV = 1138.184+1546.8,   Name = keras_1/oof_train_keras_400_0.4_200_0.2_nbags_4_nepochs_55_nfolds_5.csv
Model 4:        CV = 1133.942+1554.8,   Name = keras_2/oof_train.csv
Model 5:        CV = 1135.619+1564.4,   Name = keras_3/oof_train.csv
Model 6:        CV = 1140.481+1563.9,   Name = keras_A4/oof_train.csv
Model 7:        CV = 1138.992+1560.6,   Name = model_l2_lr_0.01_trees_7K
Model 8:        CV = 1136.702+1569.7,   Name = model_l2_bopt_run1_index75
Model 9:        CV = 1136.977+1566.1,   Name = model_l2_bopt_run2_index92
Model 10:       CV = 1135.422+1549.2,   Name = model_xgb_3
Model 11:       CV = 1146.402+1565.6,   Name = test_xgb_A1-20161103T205502
Model 12:       CV = 1133.082+1546.8,   Name = xgb_Vladimir_base_score_1-20161103T230621
Model 13:       CV = 1142.349+1552.5,   Name = xgb-1117-attempt1-20161107T234751
Model 14:       CV = 1138.632+1564.7,   Name = ./keras_A5_1/oof_train.csv
Model 15:       CV = 1136.688+1567.8,   Name = ./keras_A5_2/oof_train.csv
Model 16:       CV = 1136.114+1558.9,   Name = ./keras_A5_11/oof_train.csv
Model 17:       CV = 1135.930+1563.4,   Name = lgbm_l2_loglossshift_extremin_1112-20161110T023325
Model 18:       CV = 1132.891+1557.3,   Name = lgbm_l2_loglossshift_extremin_1112-lr_0.003-20161110T031110
Model 19:       CV = 1133.808+1549.7,   Name = lgbm_l2_loglossshift500_extremin_1112-lr_0.01-20161113T203540
Model 20:       CV = 1146.014+1587.9,   Name = ./keras_A6_1/oof_train.csv
Model 21:       CV = 1146.831+1598.9,   Name = ./keras_A6_2/oof_train.csv
Model 22:       CV = 1138.803+1570.2,   Name = ./keras_A7_1/oof_train.csv
Model 23:       CV = 1140.436+1550.8,   Name = ./keras_A8_1/oof_train.csv


# lgbm_l2_loglossshift500_extremin_1112-lr_0.01-20161113T203540.json
CV = 1133.81

**label_processor = log-shift-500**
NFOLDS = 10
        {'application': 'regression',
         'bagging_fraction': 0.8,
         'bagging_freq': 100,
         'bagging_seed': 500,
         'config': '',
         'early_stopping_round': 300,
         'exec_path': '../../../LightGBM/lightgbm',
         'feature_fraction': 0.3,
         'feature_fraction_seed': 500,
         'label_processor_function_name': 'log_shift_500',
         'learning_rate': 0.01,
         'metric': 'l1exp',
         'metric_freq': 50,
         'min_data_in_leaf': 8,
         'num_iterations': 24000,
         'num_leaves': 200,
         'num_threads': 1,
         'tree_learner': 'serial'}

---



# XGB xgb-1109-attempt_1-20161114T223803
LB = 1108.19
CV = 1131.66


# Fourth keras stacking with 25 models
LB = 1103.80
CV = 1122.53

Model 0:        CV = 1145.929+1581.4,   Name = model_fair_c_2_w_100_lr_0.002_trees_20K
Model 1:        CV = 1269.171+1569.8,   Name = model_et
Model 2:        CV = 1223.941+1693.0,   Name = skl_rf-20161108T012715
Model 3:        CV = 1138.184+1546.8,   Name = keras_1/oof_train_keras_400_0.4_200_0.2_nbags_4_nepochs_55_nfolds_5.csv
Model 4:        CV = 1133.942+1554.8,   Name = keras_2/oof_train.csv
Model 5:        CV = 1135.619+1564.4,   Name = keras_3/oof_train.csv
Model 6:        CV = 1140.481+1563.9,   Name = keras_A4/oof_train.csv
Model 7:        CV = 1138.992+1560.6,   Name = model_l2_lr_0.01_trees_7K
Model 8:        CV = 1136.702+1569.7,   Name = model_l2_bopt_run1_index75
Model 9:        CV = 1136.977+1566.1,   Name = model_l2_bopt_run2_index92
Model 10:       CV = 1135.422+1549.2,   Name = model_xgb_3
Model 11:       CV = 1146.402+1565.6,   Name = test_xgb_A1-20161103T205502
Model 12:       CV = 1133.082+1546.8,   Name = xgb_Vladimir_base_score_1-20161103T230621
Model 13:       CV = 1142.349+1552.5,   Name = xgb-1117-attempt1-20161107T234751
Model 14:       CV = 1138.632+1564.7,   Name = ./keras_A5_1/oof_train.csv
Model 15:       CV = 1136.688+1567.8,   Name = ./keras_A5_2/oof_train.csv
Model 16:       CV = 1136.114+1558.9,   Name = ./keras_A5_11/oof_train.csv
Model 17:       CV = 1135.930+1563.4,   Name = lgbm_l2_loglossshift_extremin_1112-20161110T023325
Model 18:       CV = 1132.891+1557.3,   Name = lgbm_l2_loglossshift_extremin_1112-lr_0.003-20161110T031110
Model 19:       CV = 1133.808+1549.7,   Name = lgbm_l2_loglossshift500_extremin_1112-lr_0.01-20161113T203540
Model 20:       CV = 1146.014+1587.9,   Name = ./keras_A6_1/oof_train.csv
Model 21:       CV = 1146.831+1598.9,   Name = ./keras_A6_2/oof_train.csv
Model 22:       CV = 1138.803+1570.2,   Name = ./keras_A7_1/oof_train.csv
Model 23:       CV = 1140.436+1550.8,   Name = ./keras_A8_1/oof_train.csv
Model 24:       CV = 1131.660+1547.6,   Name = xgb-1109-attempt_1-20161114T223803

Level 1 - Keras: 
        model = Sequential()
        model.add(Dense(100, input_dim = l1_x_train.shape[1], init = 'he_normal'))
        model.add(BatchNormalization())
        model.add(PReLU())
        model.add(Dropout(0.3))
        model.add(Dense(1, init = 'he_normal'))
        model.compile(loss = 'mae', optimizer = 'adam')
    nfolds = 10
    nbags = 2
    nepochs = 20
    nbags_per_fold = 2
    patience = 5


# xgb-1109-attempt_2-20161115T231535
LB = 1109.67
CV = 1127.97

**Cauchy objective on log_shift_200**
NFOLDS = 10



# xgb-1109-attempt_3-20161116T091620.json
LB = 1112.22
CV = 1133.13

**reg:linear objective on log_shift_200**
NFOLDS = 10
eta = 0.03
rounds = 640
early stopping = 100



# Fifth Keras Stacking
LB = 1103.86
CV = 1122.35

Model 0:        CV = 1145.929+1581.4,   Name = model_fair_c_2_w_100_lr_0.002_trees_20K
Model 1:        CV = 1269.171+1569.8,   Name = model_et
Model 2:        CV = 1223.941+1693.0,   Name = skl_rf-20161108T012715
Model 3:        CV = 1138.184+1546.8,   Name = keras_1/oof_train_keras_400_0.4_200_0.2_nbags_4_nepochs_55_nfolds_5.csv
Model 4:        CV = 1133.942+1554.8,   Name = keras_2/oof_train.csv
Model 5:        CV = 1135.619+1564.4,   Name = keras_3/oof_train.csv
Model 6:        CV = 1140.481+1563.9,   Name = keras_A4/oof_train.csv
Model 7:        CV = 1138.992+1560.6,   Name = model_l2_lr_0.01_trees_7K
Model 8:        CV = 1136.702+1569.7,   Name = model_l2_bopt_run1_index75
Model 9:        CV = 1136.977+1566.1,   Name = model_l2_bopt_run2_index92
Model 10:       CV = 1135.422+1549.2,   Name = model_xgb_3
Model 11:       CV = 1146.402+1565.6,   Name = test_xgb_A1-20161103T205502
Model 12:       CV = 1133.082+1546.8,   Name = xgb_Vladimir_base_score_1-20161103T230621
Model 13:       CV = 1142.349+1552.5,   Name = xgb-1117-attempt1-20161107T234751
Model 14:       CV = 1138.632+1564.7,   Name = ./keras_A5_1/oof_train.csv
Model 15:       CV = 1136.688+1567.8,   Name = ./keras_A5_2/oof_train.csv
Model 16:       CV = 1136.114+1558.9,   Name = ./keras_A5_11/oof_train.csv
Model 17:       CV = 1135.930+1563.4,   Name = lgbm_l2_loglossshift_extremin_1112-20161110T023325
Model 18:       CV = 1132.891+1557.3,   Name = lgbm_l2_loglossshift_extremin_1112-lr_0.003-20161110T031110
Model 19:       CV = 1133.808+1549.7,   Name = lgbm_l2_loglossshift500_extremin_1112-lr_0.01-20161113T203540
Model 20:       CV = 1146.014+1587.9,   Name = ./keras_A6_1/oof_train.csv
Model 21:       CV = 1146.831+1598.9,   Name = ./keras_A6_2/oof_train.csv
Model 22:       CV = 1138.803+1570.2,   Name = ./keras_A7_1/oof_train.csv
Model 23:       CV = 1140.436+1550.8,   Name = ./keras_A8_1/oof_train.csv
Model 24:       CV = 1131.660+1547.6,   Name = xgb-1109-attempt_1-20161114T223803
Model 25:       CV = 1127.974+1552.0,   Name = xgb-1109-attempt_2-20161115T231535
Model 26:       CV = 1133.127+1549.4,   Name = xgb-1109-attempt_3-20161116T091620

Level 1 - Keras: 
        model = Sequential()
        model.add(Dense(100, input_dim = l1_x_train.shape[1], init = 'he_normal'))
        model.add(BatchNormalization())
        model.add(PReLU())
        model.add(Dropout(0.3))
        model.add(Dense(1, init = 'he_normal'))
        model.compile(loss = 'mae', optimizer = 'adam')
    nfolds = 10
    nbags = 2
    nepochs = 20
    nbags_per_fold = 2
    patience = 5


# Sixth keras stacking
LB = 1103.54
CV = 1122.026

nfolds = 10
nbags = 4
nepochs = 20
nbags_per_fold = 1


Model 0:        CV = 1145.929+1581.4,   Name = model_fair_c_2_w_100_lr_0.002_trees_20K
Model 1:        CV = 1269.171+1569.8,   Name = model_et
Model 2:        CV = 1223.941+1693.0,   Name = skl_rf-20161108T012715
Model 3:        CV = 1138.184+1546.8,   Name = keras_1/oof_train_keras_400_0.4_200_0.2_nbags_4_nepochs_55_nfolds_5.csv
Model 4:        CV = 1133.942+1554.8,   Name = keras_2/oof_train.csv
Model 5:        CV = 1135.619+1564.4,   Name = keras_3/oof_train.csv
Model 6:        CV = 1140.481+1563.9,   Name = keras_A4/oof_train.csv
Model 7:        CV = 1138.992+1560.6,   Name = model_l2_lr_0.01_trees_7K
Model 8:        CV = 1136.702+1569.7,   Name = model_l2_bopt_run1_index75
Model 9:        CV = 1136.977+1566.1,   Name = model_l2_bopt_run2_index92
Model 10:       CV = 1135.422+1549.2,   Name = model_xgb_3
Model 11:       CV = 1146.402+1565.6,   Name = test_xgb_A1-20161103T205502
Model 12:       CV = 1133.082+1546.8,   Name = xgb_Vladimir_base_score_1-20161103T230621
Model 13:       CV = 1142.349+1552.5,   Name = xgb-1117-attempt1-20161107T234751
Model 14:       CV = 1138.632+1564.7,   Name = ./keras_A5_1/oof_train.csv
Model 15:       CV = 1136.688+1567.8,   Name = ./keras_A5_2/oof_train.csv
Model 16:       CV = 1136.114+1558.9,   Name = ./keras_A5_11/oof_train.csv
Model 17:       CV = 1135.930+1563.4,   Name = lgbm_l2_loglossshift_extremin_1112-20161110T023325
Model 18:       CV = 1132.891+1557.3,   Name = lgbm_l2_loglossshift_extremin_1112-lr_0.003-20161110T031110
Model 19:       CV = 1133.808+1549.7,   Name = lgbm_l2_loglossshift500_extremin_1112-lr_0.01-20161113T203540
Model 20:       CV = 1146.014+1587.9,   Name = ./keras_A6_1/oof_train.csv
Model 21:       CV = 1146.831+1598.9,   Name = ./keras_A6_2/oof_train.csv
Model 22:       CV = 1138.803+1570.2,   Name = ./keras_A7_1/oof_train.csv
Model 23:       CV = 1140.436+1550.8,   Name = ./keras_A8_1/oof_train.csv
Model 24:       CV = 1131.660+1547.6,   Name = xgb-1109-attempt_1-20161114T223803
Model 25:       CV = 1127.974+1552.0,   Name = xgb-1109-attempt_2-20161115T231535
Model 26:       CV = 1133.127+1549.4,   Name = xgb-1109-attempt_3-20161116T091620
Model 27:       CV = 1219.379+1626.8,   Name = skl_Ridge-quad-100020161117T103454
Model 28:       CV = 1216.512+1629.9,   Name = skl_Ridge-quad-100020161117T104911
Model 29:       CV = 1216.830+1647.6,   Name = skl_Ridge-quad-100020161117T110254
Model 30:       CV = 1220.054+1627.3,   Name = skl_Ridge-quad-100020161117T111651
Model 31:       CV = 1217.454+1627.0,   Name = skl_Ridge-quad-100020161117T113232
Model 32:       CV = 1216.311+1705.7,   Name = skl_Ridge-quad-100020161117T114551
Model 33:       CV = 1217.437+1625.2,   Name = skl_Ridge-quad-100020161117T120118
Model 34:       CV = 1221.585+1634.2,   Name = skl_Ridge-quad-100020161117T121525
Model 35:       CV = 1220.311+1654.0,   Name = skl_Ridge-quad-100020161117T123053
Model 36:       CV = 1226.223+1728.5,   Name = skl_Ridge-quad-100020161117T124430


# xgb-1109-attempt_4-base_score_7.7-20161119T202505 
LB = 1114.29
CV = 1136.61

NFOLDS = 10
**COMB_features not used?**
{**'base_score': 7.747,**
 'colsample_bytree': 0.7,
 'early_stopping_rounds': 100,
 'label_processor_function_name': 'log_shift_200',
 'learning_rate': 0.03,
 'max_depth': 12,
 'min_child_weight': 100,
 'nrounds': 640,
 'seed': 13579,
 'subsample': 0.7,
 'verbose_eval': 10}

# xgb-1109-attempt_5-preprocessor_pow_0.25_shift_1-20161119T225103
LB = 1114.41
CV = 1134.47
**COMB_features not used!!**

NFOLDS = 10
{'base_score': 1,
 'colsample_bytree': 0.7,
 'early_stopping_rounds': 100,
 **'label_processor_function_name': 'pow_0.25_shift_1',**
 'learning_rate': 0.03,
 'max_depth': 12,
 'min_child_weight': 100,
 'nrounds': 640,
 'seed': 13579,
 'subsample': 0.7,
 'verbose_eval': 10}


# xgb-1109-attempt_5-preprocessor_pow_0.2_shift_10-20161120T055959
LB = 1113.70
CV = 1134.35
**COMB_features not used!!**

'base_score': 1,
 'booster': 'gbtree',
 'colsample_bytree': 0.7,
 'early_stopping_rounds': 100,
 **'label_processor_function_name': 'pow_0.2_shift_10',**
 'learning_rate': 0.03,
 'max_depth': 12,
 'min_child_weight': 100,
 'nrounds': 640,
 'seed': 13579,
 'subsample': 0.7,
 'verbose_eval': 10}
 
# xgb-1109-attempt_6-logregobj-preprocessor_pow_0.2_shift_10-20161120T112441
CV = 1156.34

**objective=logregobj**
**COMB_features not used!!**

NFOLDS = 10
{'base_score': 1,
 'booster': 'gbtree',
 'colsample_bytree': 0.7,
 **'early_stopping_rounds': 25,**
 'label_processor_function_name': 'pow_0.2_shift_10',
 'learning_rate': 0.03,
 'max_depth': 12,
 'min_child_weight': 100,
 'nrounds': 640,
 'seed': 1379,
 'subsample': 0.7,
 'verbose_eval': 10}

# ALL xgb-1109-attempt_6

**Not included in stacking**

xgb-1109-attempt_6-logregobj-preprocessor_pow_0.2_shift_10-20161120T073536.json
CV =  1157.01531263

xgb-1109-attempt_6-logregobj-preprocessor_pow_0.2_shift_10-20161120T112441.json
CV =  1156.34533948

xgb-1109-attempt_6-logregobj-preprocessor_log_shift_200-20161120T121531.json
CV =  1159.78776911

xgb-1109-attempt_6-logregobj-preprocessor_log_shift_200-20161120T141649.json
CV =  1159.69772797

{'seed': 1379, 'subsample': 0.7, 'early_stopping_rounds': 100, 'nrounds': 640, 'silent': 1, 'booster': 'gbtree', 'label_processor_function_name': 'pow_0.2_shift_10', 'min_child_weight': 100, 'base_score': 1, 'verbose_eval': 10, 'colsample_bytree': 0.7, 'learning_rate': 0.03, 'max_depth': 12}
{'seed': 1379, 'subsample': 0.7, 'early_stopping_rounds': 25,  'nrounds': 640, 'silent': 1, 'booster': 'gbtree', 'label_processor_function_name': 'pow_0.2_shift_10', 'min_child_weight': 100, 'base_score': 1, 'verbose_eval': 10, 'colsample_bytree': 0.7, 'learning_rate': 0.03, 'max_depth': 12}
{'seed': 1579, 'subsample': 0.7, 'early_stopping_rounds': 25,  'nrounds': 640, 'silent': 1, 'booster': 'gbtree', 'label_processor_function_name': 'log_shift_200',    'min_child_weight': 100, 'base_score': 1, 'verbose_eval': 10, 'colsample_bytree': 0.7, 'learning_rate': 0.03, 'max_depth': 12}
{'seed': 1579, 'subsample': 0.7, 'early_stopping_rounds': 100, 'nrounds': 640, 'silent': 1, 'booster': 'gbtree', 'label_processor_function_name': 'log_shift_200',    'min_child_weight': 100, 'base_score': 1, 'verbose_eval': 10, 'colsample_bytree': 0.7, 'learning_rate': 0.03, 'max_depth': 12}


# xgb-1109-attempt_7-cauchy-preprocessor_log_shift_200-20161120T151359
CV = 1127.93

**Repeat of xgb-1109-attempt_2!**

**COMB_features used with cauchy objective**

NFOLDS = 10
{'base_score': 1,
 'booster': 'gbtree',
 'colsample_bytree': 0.7,
 'early_stopping_rounds': 100,
 'label_processor_function_name': 'log_shift_200',
 'learning_rate': 0.03,
 'max_depth': 12,
 'min_child_weight': 100,
 'nrounds': 640,
 'seed': 1579,
 'subsample': 0.7,
 'verbose_eval': 10}


# xgb-1109-attempt_8-obj_reglinear-preprocessor_pow_0.2_shift_10-20161120T225122
LB = 1112.66
CV = 1131.25


**COMB features used with reg:linear objective**
NFOLDS = 10
{'base_score': 1,
 'booster': 'gbtree',
 'colsample_bytree': 0.7,
 'early_stopping_rounds': 100,
 **'label_processor_function_name': 'pow_0.2_shift_10',**
 'learning_rate': 0.03,
 'max_depth': 12,
 'min_child_weight': 100,
 'nrounds': 640,
 'seed': ?,
 'subsample': 0.7,
 'verbose_eval': 10}


# xgb-1109-attempt_9-obj_reglinear-preprocessor_log_shift_200-20161121T092757
CV = 1140.10

**1000 sampled COMB features used wuth reg:linear objective**

NFOLDS = 10
{'base_score': 1,
 'booster': 'gbtree',
 'colsample_bytree': 0.7,
 'early_stopping_rounds': 50,
 **'label_processor_function_name': 'log_shift_200',**
 'learning_rate': 0.03,
 'max_depth': 12,
 'min_child_weight': 100,
 'nrounds': 640,
 'seed': 3579,
 'subsample': 0.7,
 'verbose_eval': 10}
 
 
# Seventh keras stacking
LB = 1103.84
CV = 1121.763

nfolds = 10
nbags = 4
nepochs = 20
nbags_per_fold = 1

Model 0:        CV = 1145.929+1581.4,   Name = model_fair_c_2_w_100_lr_0.002_trees_20K
Model 1:        CV = 1269.171+1569.8,   Name = model_et
Model 2:        CV = 1223.941+1693.0,   Name = skl_rf-20161108T012715
Model 3:        CV = 1138.184+1546.8,   Name = keras_1/oof_train_keras_400_0.4_200_0.2_nbags_4_nepochs_55_nfolds_5.csv
Model 4:        CV = 1133.942+1554.8,   Name = keras_2/oof_train.csv
Model 5:        CV = 1135.619+1564.4,   Name = keras_3/oof_train.csv
Model 6:        CV = 1140.481+1563.9,   Name = keras_A4/oof_train.csv
Model 7:        CV = 1138.992+1560.6,   Name = model_l2_lr_0.01_trees_7K
Model 8:        CV = 1136.702+1569.7,   Name = model_l2_bopt_run1_index75
Model 9:        CV = 1136.977+1566.1,   Name = model_l2_bopt_run2_index92
Model 10:       CV = 1135.422+1549.2,   Name = model_xgb_3
Model 11:       CV = 1146.402+1565.6,   Name = test_xgb_A1-20161103T205502
Model 12:       CV = 1133.082+1546.8,   Name = xgb_Vladimir_base_score_1-20161103T230621
Model 13:       CV = 1142.349+1552.5,   Name = xgb-1117-attempt1-20161107T234751
Model 14:       CV = 1138.632+1564.7,   Name = ./keras_A5_1/oof_train.csv
Model 15:       CV = 1136.688+1567.8,   Name = ./keras_A5_2/oof_train.csv
Model 16:       CV = 1136.114+1558.9,   Name = ./keras_A5_11/oof_train.csv
Model 17:       CV = 1135.930+1563.4,   Name = lgbm_l2_loglossshift_extremin_1112-20161110T023325
Model 18:       CV = 1132.891+1557.3,   Name = lgbm_l2_loglossshift_extremin_1112-lr_0.003-20161110T031110
Model 19:       CV = 1133.808+1549.7,   Name = lgbm_l2_loglossshift500_extremin_1112-lr_0.01-20161113T203540
Model 20:       CV = 1146.014+1587.9,   Name = ./keras_A6_1/oof_train.csv
Model 21:       CV = 1146.831+1598.9,   Name = ./keras_A6_2/oof_train.csv
Model 22:       CV = 1138.803+1570.2,   Name = ./keras_A7_1/oof_train.csv
Model 23:       CV = 1140.436+1550.8,   Name = ./keras_A8_1/oof_train.csv
Model 24:       CV = 1131.660+1547.6,   Name = xgb-1109-attempt_1-20161114T223803
Model 25:       CV = 1127.974+1552.0,   Name = xgb-1109-attempt_2-20161115T231535
Model 26:       CV = 1133.127+1549.4,   Name = xgb-1109-attempt_3-20161116T091620
Model 27:       CV = 1219.379+1626.8,   Name = skl_Ridge-quad-100020161117T103454
Model 28:       CV = 1216.512+1629.9,   Name = skl_Ridge-quad-100020161117T104911
Model 29:       CV = 1216.830+1647.6,   Name = skl_Ridge-quad-100020161117T110254
Model 30:       CV = 1220.054+1627.3,   Name = skl_Ridge-quad-100020161117T111651
Model 31:       CV = 1217.454+1627.0,   Name = skl_Ridge-quad-100020161117T113232
Model 32:       CV = 1216.311+1705.7,   Name = skl_Ridge-quad-100020161117T114551
Model 33:       CV = 1217.437+1625.2,   Name = skl_Ridge-quad-100020161117T120118
Model 34:       CV = 1221.585+1634.2,   Name = skl_Ridge-quad-100020161117T121525
Model 35:       CV = 1220.311+1654.0,   Name = skl_Ridge-quad-100020161117T123053
Model 36:       CV = 1226.223+1728.5,   Name = skl_Ridge-quad-100020161117T124430
Model 37:       CV = 1136.613+1554.2,   Name = xgb-1109-attempt_4-base_score_7.7-20161119T202505
Model 38:       CV = 1134.472+1529.0,   Name = xgb-1109-attempt_5-preprocessor_pow_0.25_shift_1-20161119T225103
Model 39:       CV = 1131.255+1532.5,   Name = xgb-1109-attempt_8-obj_reglinear-preprocessor_pow_0.2_shift_10-20161120T225122
Model 40:       CV = 1140.101+1558.0,   Name = xgb-1109-attempt_9-obj_reglinear-preprocessor_log_shift_200-20161121T092757
(188318, 41),(125546, 41)


# xgb-mrooijer_1130-attempt_1-obj_fair1.5-preprocessor_pow_0.25_shift_1-base_score-8.4738505029380491-20161121T215220
LB = 1106.67
CV = 1129.31

{'base_score': 8.4738505029380491,
 'booster': 'gbtree',
 'colsample_bytree': 0.08,
 'early_stopping_rounds': 50,
 'label_processor_function_name': 'pow_0.25_shift_1',
 'learning_rate': 0.05,
 'max_depth': 12,
 'min_child_weight': 100,
 'nrounds': 640,
 'seed': 3579,
 'silent': 1,
 'subsample': 0.7,
 'verbose_eval': 10}


# xgb-mrooijer_1130-attempt_2-obj_fair1.5-preprocessor_pow_0.25_shift_1-base_score-8.4738505029380491-20161121T232803
LB = 1105.28
CV = 1124.92

{'base_score': 8.4738505029380491,
 'booster': 'gbtree',
 'colsample_bytree': 0.08,
 'early_stopping_rounds': 100,
 'label_processor_function_name': 'pow_0.25_shift_1',
 **'learning_rate': 0.01,**
 'max_depth': 12,
 'min_child_weight': 100,
 'nrounds': 3500,
 'seed': 3579,
 'silent': 1,
 'subsample': 0.7,
 'verbose_eval': 50}
 
 
# Eighth Keras Stacking
LB = 1101.91
CV = 1120.62

nfolds = 10
nbags = 4
nepochs = 20
nbags_per_fold = 1

Model 0:        CV = 1145.929+1581.4,   Name = model_fair_c_2_w_100_lr_0.002_trees_20K 
Model 1:        CV = 1269.171+1569.8,   Name = model_et 
Model 2:        CV = 1223.941+1693.0,   Name = skl_rf-20161108T012715 
Model 3:        CV = 1138.184+1546.8,   Name = keras_1/oof_train_keras_400_0.4_200_0.2_nbags_4_nepochs_55_nfolds_5.csv 
Model 4:        CV = 1133.942+1554.8,   Name = keras_2/oof_train.csv 
Model 5:        CV = 1135.619+1564.4,   Name = keras_3/oof_train.csv 
Model 6:        CV = 1140.481+1563.9,   Name = keras_A4/oof_train.csv 
Model 7:        CV = 1138.992+1560.6,   Name = model_l2_lr_0.01_trees_7K 
Model 8:        CV = 1136.702+1569.7,   Name = model_l2_bopt_run1_index75 
Model 9:        CV = 1136.977+1566.1,   Name = model_l2_bopt_run2_index92 
Model 10:       CV = 1135.422+1549.2,   Name = model_xgb_3 
Model 11:       CV = 1146.402+1565.6,   Name = test_xgb_A1-20161103T205502 
Model 12:       CV = 1133.082+1546.8,   Name = xgb_Vladimir_base_score_1-20161103T230621 
Model 13:       CV = 1142.349+1552.5,   Name = xgb-1117-attempt1-20161107T234751 
Model 14:       CV = 1138.632+1564.7,   Name = ./keras_A5_1/oof_train.csv 
Model 15:       CV = 1136.688+1567.8,   Name = ./keras_A5_2/oof_train.csv 
Model 16:       CV = 1136.114+1558.9,   Name = ./keras_A5_11/oof_train.csv 
Model 17:       CV = 1135.930+1563.4,   Name = lgbm_l2_loglossshift_extremin_1112-20161110T023325 
Model 18:       CV = 1132.891+1557.3,   Name = lgbm_l2_loglossshift_extremin_1112-lr_0.003-20161110T031110 
Model 19:       CV = 1133.808+1549.7,   Name = lgbm_l2_loglossshift500_extremin_1112-lr_0.01-20161113T203540 
Model 20:       CV = 1146.014+1587.9,   Name = ./keras_A6_1/oof_train.csv 
Model 21:       CV = 1146.831+1598.9,   Name = ./keras_A6_2/oof_train.csv 
Model 22:       CV = 1138.803+1570.2,   Name = ./keras_A7_1/oof_train.csv 
Model 23:       CV = 1140.436+1550.8,   Name = ./keras_A8_1/oof_train.csv 
Model 24:       CV = 1131.660+1547.6,   Name = xgb-1109-attempt_1-20161114T223803 
Model 25:       CV = 1127.974+1552.0,   Name = xgb-1109-attempt_2-20161115T231535 
Model 26:       CV = 1133.127+1549.4,   Name = xgb-1109-attempt_3-20161116T091620 
Model 27:       CV = 1219.379+1626.8,   Name = skl_Ridge-quad-100020161117T103454 
Model 28:       CV = 1216.512+1629.9,   Name = skl_Ridge-quad-100020161117T104911 
Model 29:       CV = 1216.830+1647.6,   Name = skl_Ridge-quad-100020161117T110254 
Model 30:       CV = 1220.054+1627.3,   Name = skl_Ridge-quad-100020161117T111651 
Model 31:       CV = 1217.454+1627.0,   Name = skl_Ridge-quad-100020161117T113232 
Model 32:       CV = 1216.311+1705.7,   Name = skl_Ridge-quad-100020161117T114551 
Model 33:       CV = 1217.437+1625.2,   Name = skl_Ridge-quad-100020161117T120118 
Model 34:       CV = 1221.585+1634.2,   Name = skl_Ridge-quad-100020161117T121525 
Model 35:       CV = 1220.311+1654.0,   Name = skl_Ridge-quad-100020161117T123053 
Model 36:       CV = 1226.223+1728.5,   Name = skl_Ridge-quad-100020161117T124430 
Model 37:       CV = 1136.613+1554.2,   Name = xgb-1109-attempt_4-base_score_7.7-20161119T202505 
Model 38:       CV = 1134.472+1529.0,   Name = xgb-1109-attempt_5-preprocessor_pow_0.25_shift_1-20161119T225103 
Model 39:       CV = 1131.255+1532.5,   Name = xgb-1109-attempt_8-obj_reglinear-preprocessor_pow_0.2_shift_10-20161120T225122 
Model 40:       CV = 1140.101+1558.0,   Name = xgb-1109-attempt_9-obj_reglinear-preprocessor_log_shift_200-20161121T092757 
Model 41:       CV = 1129.313+1545.3,   Name = xgb-mrooijer_1130-attempt_1-obj_fair1.5-preprocessor_pow_0.25_shift_1-base_score-8.4738505029380491-20161121T215220 
Model 42:       CV = 1124.922+1540.4,   Name = xgb-mrooijer_1130-attempt_2-obj_fair1.5-preprocessor_pow_0.25_shift_1-base_score-8.4738505029380491-20161121T232803 
(188318, 43),(125546, 43)


# xgb-mrooijer-lin-attempt_1-obj_fair1.5-preprocessor_pow_0.25_shift_1-base_score-8.4738505029380491-20161123T001744
LB = 1105.07
CV = 1128.91

# xgb-mrooijer-lin-attempt_2-obj_fair1.5-preprocessor_pow_0.25_shift_1-base_score-8.4738505029380491-20161123T013341
**Lowered eta to 0.01**

LB = 1104.68
CV = 1124.52


# Ninth Keras Stacking 
46 models
Added :
xgb-1109-attempt_5-preprocessor_pow_0.2_shift_10-20161120T055959
xgb-mrooijer-lin-attempt_1-obj_fair1.5-preprocessor_pow_0.25_shift_1-base_score-8.4738505029380491-20161123T001744
xgb-mrooijer-lin-attempt_2-obj_fair1.5-preprocessor_pow_0.25_shift_1-base_score-8.4738505029380491-20161123T013341

First Run
LB = 1101.85
CV = 1121.06

Second Run
LB = 1101.67
CV = 1120.62

Third Run
LB = did-not-submit
CV = 1121.15


# xgb-mrooijer_1130-attempt_3-obj_fair1.5-preprocessor_pow_0.25_shift_1-base_score-8.4738505029380491-20161122T095124
**eta back to 0.05**
**lower colsample_by_tree to 0.035**

CV = 1138.4
**Needs more rounds**


# xgb-mrooijer_1130-attempt_4-obj_fair1.5-preprocessor_pow_0.25_shift_1-base_score-8.4738505029380491-20161122T200929
**attempt_1 + max_depth increased from 12 to 14**
CV = 1130.04


# xgb-mrooijer_1130-attempt_5-obj_fair0.7-preprocessor_pow_0.25_shift_1-base_score-8.4738505029380491-20161122T214600
LB = 1107.74
CV = 1129.41

{'base_score': 8.473850502938049,
 'booster': 'gbtree',
 'colsample_bytree': 0.08,
 'early_stopping_rounds': 50,
 'label_processor_function_name': 'pow_0.25_shift_1',
 'learning_rate': 0.05,
 'max_depth': 12,
 'min_child_weight': 100,
 'nrounds': 640,
 'seed': 3579,
 'silent': 1,
 'subsample': 0.7,
 'verbose_eval': 10}
 
 
# Tenth Keras Stacking
LB = 1101.32
CV = 1119.437

**Included all cont features + linearized cont features + all cat features + top 5 cat features + 47 models**
Model 0:        CV = 1145.929+1581.4,   Name = model_fair_c_2_w_100_lr_0.002_trees_20K
Model 1:        CV = 1269.171+1569.8,   Name = model_et
Model 2:        CV = 1223.941+1693.0,   Name = skl_rf-20161108T012715
Model 3:        CV = 1138.184+1546.8,   Name = keras_1/oof_train_keras_400_0.4_200_0.2_nbags_4_nepochs_55_nfolds_5.csv
Model 4:        CV = 1133.942+1554.8,   Name = keras_2/oof_train.csv
Model 5:        CV = 1135.619+1564.4,   Name = keras_3/oof_train.csv
Model 6:        CV = 1140.481+1563.9,   Name = keras_A4/oof_train.csv
Model 7:        CV = 1138.992+1560.6,   Name = model_l2_lr_0.01_trees_7K
Model 8:        CV = 1136.702+1569.7,   Name = model_l2_bopt_run1_index75
Model 9:        CV = 1136.977+1566.1,   Name = model_l2_bopt_run2_index92
Model 10:       CV = 1135.422+1549.2,   Name = model_xgb_3
Model 11:       CV = 1146.402+1565.6,   Name = test_xgb_A1-20161103T205502
Model 12:       CV = 1133.082+1546.8,   Name = xgb_Vladimir_base_score_1-20161103T230621
Model 13:       CV = 1142.349+1552.5,   Name = xgb-1117-attempt1-20161107T234751
Model 14:       CV = 1138.632+1564.7,   Name = ./keras_A5_1/oof_train.csv
Model 15:       CV = 1136.688+1567.8,   Name = ./keras_A5_2/oof_train.csv
Model 16:       CV = 1136.114+1558.9,   Name = ./keras_A5_11/oof_train.csv
Model 17:       CV = 1135.930+1563.4,   Name = lgbm_l2_loglossshift_extremin_1112-20161110T023325
Model 18:       CV = 1132.891+1557.3,   Name = lgbm_l2_loglossshift_extremin_1112-lr_0.003-20161110T031110
Model 19:       CV = 1133.808+1549.7,   Name = lgbm_l2_loglossshift500_extremin_1112-lr_0.01-20161113T203540
Model 20:       CV = 1146.014+1587.9,   Name = ./keras_A6_1/oof_train.csv
Model 21:       CV = 1146.831+1598.9,   Name = ./keras_A6_2/oof_train.csv
Model 22:       CV = 1138.803+1570.2,   Name = ./keras_A7_1/oof_train.csv
Model 23:       CV = 1140.436+1550.8,   Name = ./keras_A8_1/oof_train.csv
Model 24:       CV = 1131.660+1547.6,   Name = xgb-1109-attempt_1-20161114T223803
Model 25:       CV = 1127.974+1552.0,   Name = xgb-1109-attempt_2-20161115T231535
Model 26:       CV = 1133.127+1549.4,   Name = xgb-1109-attempt_3-20161116T091620
Model 27:       CV = 1219.379+1626.8,   Name = skl_Ridge-quad-100020161117T103454
Model 28:       CV = 1216.512+1629.9,   Name = skl_Ridge-quad-100020161117T104911
Model 29:       CV = 1216.830+1647.6,   Name = skl_Ridge-quad-100020161117T110254
Model 30:       CV = 1220.054+1627.3,   Name = skl_Ridge-quad-100020161117T111651
Model 31:       CV = 1217.454+1627.0,   Name = skl_Ridge-quad-100020161117T113232
Model 32:       CV = 1216.311+1705.7,   Name = skl_Ridge-quad-100020161117T114551
Model 33:       CV = 1217.437+1625.2,   Name = skl_Ridge-quad-100020161117T120118
Model 34:       CV = 1221.585+1634.2,   Name = skl_Ridge-quad-100020161117T121525
Model 35:       CV = 1220.311+1654.0,   Name = skl_Ridge-quad-100020161117T123053
Model 36:       CV = 1226.223+1728.5,   Name = skl_Ridge-quad-100020161117T124430
Model 37:       CV = 1136.613+1554.2,   Name = xgb-1109-attempt_4-base_score_7.7-20161119T202505
Model 38:       CV = 1134.472+1529.0,   Name = xgb-1109-attempt_5-preprocessor_pow_0.25_shift_1-20161119T225103
Model 39:       CV = 1134.353+1533.7,   Name = xgb-1109-attempt_5-preprocessor_pow_0.2_shift_10-20161120T055959
Model 40:       CV = 1131.255+1532.5,   Name = xgb-1109-attempt_8-obj_reglinear-preprocessor_pow_0.2_shift_10-20161120T225122
Model 41:       CV = 1140.101+1558.0,   Name = xgb-1109-attempt_9-obj_reglinear-preprocessor_log_shift_200-20161121T092757
Model 42:       CV = 1129.313+1545.3,   Name = xgb-mrooijer_1130-attempt_1-obj_fair1.5-preprocessor_pow_0.25_shift_1-base_score-8.4738505029380491-20161121T215220
Model 43:       CV = 1124.922+1540.4,   Name = xgb-mrooijer_1130-attempt_2-obj_fair1.5-preprocessor_pow_0.25_shift_1-base_score-8.4738505029380491-20161121T232803
Model 44:       CV = 1129.411+1551.2,   Name = xgb-mrooijer_1130-attempt_5-obj_fair0.7-preprocessor_pow_0.25_shift_1-base_score-8.4738505029380491-20161122T214600
Model 45:       CV = 1128.918+1541.5,   Name = xgb-mrooijer-lin-attempt_1-obj_fair1.5-preprocessor_pow_0.25_shift_1-base_score-8.4738505029380491-20161123T001744
Model 46:       CV = 1124.523+1536.9,   Name = xgb-mrooijer-lin-attempt_2-obj_fair1.5-preprocessor_pow_0.25_shift_1-base_score-8.4738505029380491-20161123T013341

(188318, 196),(125546, 196)


nfolds = 10
**nbags = 10**
nepochs = 20
nbags_per_fold = 1


# Eleven Keras Stacking
LB = 1101.316
CV = 1119.338 + std 9.98

47 models
212 features

all features + lin_features + cat80,cat87,cat79,cat89 as dummies

nfolds = 10
**nbags = 80**
nepochs = 20
nbags_per_fold = 1
**Run time: 1d6h**

 
 
# Keras_A9_1
LB = 1120.88
CV = 1138.82


**3 hidden NN + 10 fold + lin_cont + pow_0.25_shift_1_basescore_7.94**

        def nn_model():
            model = Sequential()
            model.add(Dense(400, input_dim = xtrain.shape[1], init = 'he_normal'))
            model.add(BatchNormalization())
            model.add(PReLU())
            model.add(Dropout(0.5))
            model.add(Dense(200, init = 'he_normal'))
            model.add(BatchNormalization())
            model.add(PReLU())
            model.add(Dropout(0.4))
            model.add(Dense(50, init = 'he_normal'))
            # model.add(Dense(100, init = 'he_normal'))
            model.add(BatchNormalization())
            model.add(PReLU())
            model.add(Dropout(0.3))
            model.add(Dense(1, init = 'he_normal'))
            model.add(Lambda(lambda x: x + 6.94 + 1.0))
            model.add(Lambda(lambda x: x**4))
            model.add(Lambda(lambda x: x-1))
            model.compile(loss = 'mae', optimizer = 'adam')
            return(model)

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=2, verbose=0, mode='auto'),
            ]
nfolds = 10
nepoch = 200
pateince = 2
nbags = 4

Run time 3.5h
Avg epoch ~ 17


# Keras_A9_2
LB = 1115.63
CV = 1135.29

**keras_2 + lin_cont + pow_0.25_shift_1_basescore_7.94**

def nn_model():
    model = Sequential()
    model.add(Dense(400, input_dim = xtrain.shape[1], init = 'he_normal'))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(Dropout(0.4))
    
    model.add(Dense(200, init = 'he_normal'))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(Dropout(0.2))
    
    model.add(Dense(50, init = 'he_normal'))
    #  model.add(Dense(100, init = 'he_normal'))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(Dropout(0.2))
    
    model.add(Dense(1, init = 'he_normal'))
    model.add(Lambda(lambda x: x + 6.94 + 1.0))
    model.add(Lambda(lambda x: x**4))
    model.add(Lambda(lambda x: x-1))
    model.compile(loss = 'mae', optimizer = 'adam')
    return(model)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto'),
    ]

**nfolds = 5**
**nbags = 5**
nepochs = 200
**patience = 5**
Batch size may have been 512

Run-time 2h18m


# Keras_A9_3
LB = 1121.07
CV = 1114.77

With ModelCheckpoint
**ModelCheckpoint did not work correctly, did not reset best val_loss across folds or bags**

nfolds = 5
nbags = 5
nepochs = 200
patience = 5
Batch size 256


# Keras_A9_4
LB = 1116.51
CV = 1132.96

**With correct ModelCheckpoint**

nfolds = 5
nbags = 5
nepochs = 200
patience = 5
Batch size 256


# Twelveth Keras Stacking
LB = 1104.91
CV = 1103.925 + std-10-fold 8.0

51 models
216 features

all features + lin_features + cat80,cat87,cat79,cat89 as dummies

nfolds = 10
nbags = 10
nepochs = 20
nbags_per_fold = 1
Run time:  3h45m

Model 0:        CV = 1145.929+1581.4,   Name = model_fair_c_2_w_100_lr_0.002_trees_20K
...
Model 35:       CV = 1220.311+1654.0,   Name = skl_Ridge-quad-100020161117T123053
Model 36:       CV = 1226.223+1728.5,   Name = skl_Ridge-quad-100020161117T124430
Model 37:       CV = 1136.613+1554.2,   Name = xgb-1109-attempt_4-base_score_7.7-20161119T202505
Model 38:       CV = 1134.472+1529.0,   Name = xgb-1109-attempt_5-preprocessor_pow_0.25_shift_1-20161119T225103
Model 39:       CV = 1134.353+1533.7,   Name = xgb-1109-attempt_5-preprocessor_pow_0.2_shift_10-20161120T055959
Model 40:       CV = 1131.255+1532.5,   Name = xgb-1109-attempt_8-obj_reglinear-preprocessor_pow_0.2_shift_10-20161120T225122
Model 41:       CV = 1140.101+1558.0,   Name = xgb-1109-attempt_9-obj_reglinear-preprocessor_log_shift_200-20161121T092757
Model 42:       CV = 1129.313+1545.3,   Name = xgb-mrooijer_1130-attempt_1-obj_fair1.5-preprocessor_pow_0.25_shift_1-base_score-8.4738505029380491-20161121T215220
Model 43:       CV = 1124.922+1540.4,   Name = xgb-mrooijer_1130-attempt_2-obj_fair1.5-preprocessor_pow_0.25_shift_1-base_score-8.4738505029380491-20161121T232803
Model 44:       CV = 1129.411+1551.2,   Name = xgb-mrooijer_1130-attempt_5-obj_fair0.7-preprocessor_pow_0.25_shift_1-base_score-8.4738505029380491-20161122T214600
Model 45:       CV = 1128.918+1541.5,   Name = xgb-mrooijer-lin-attempt_1-obj_fair1.5-preprocessor_pow_0.25_shift_1-base_score-8.4738505029380491-20161123T001744
Model 46:       CV = 1124.523+1536.9,   Name = xgb-mrooijer-lin-attempt_2-obj_fair1.5-preprocessor_pow_0.25_shift_1-base_score-8.4738505029380491-20161123T013341
Model 47:       CV = 1138.821+1565.0,   Name = ./keras_A9_1/oof_train.csv
Model 48:       CV = 1135.291+1531.5,   Name = ./keras_A9_2/oof_train.csv
Model 49:       CV = 1114.779+1513.6,   Name = ./keras_A9_3/oof_train.csv
Model 50:       CV = 1132.961+1530.5,   Name = ./keras_A9_4/oof_train.csv
(188318, 216),(125546, 216)

        model = Sequential()
        model.add(Dense(100, input_dim = l1_x_train.shape[1], init = 'he_normal'))
        model.add(BatchNormalization())
        model.add(PReLU())
        model.add(Dropout(0.3))
        model.add(Dense(1, init = 'he_normal'))
        model.compile(loss = 'mae', optimizer = 'adam')
        return(model)

**keras_A9_3 leaks information**



# Twelve-B Keras Stacking
LB = 1101.21
CV = 1119.206 + std-10-fold 6.5

50 models
215 features

all features + lin_features + top-cats as dummies

nfolds = 10
nbags = 10
nepochs = 20
nbags_per_fold = 1
Run time:  3h51m

Model 0:        CV = 1145.929+1581.4,   Name = model_fair_c_2_w_100_lr_0.002_trees_20K
Model 1:        CV = 1269.171+1569.8,   Name = model_et
Model 2:        CV = 1223.941+1693.0,   Name = skl_rf-20161108T012715
Model 3:        CV = 1138.184+1546.8,   Name = keras_1/oof_train_keras_400_0.4_200_0.2_nbags_4_nepochs_55_nfolds_5.csv
Model 4:        CV = 1133.942+1554.8,   Name = keras_2/oof_train.csv
Model 5:        CV = 1135.619+1564.4,   Name = keras_3/oof_train.csv
Model 6:        CV = 1140.481+1563.9,   Name = keras_A4/oof_train.csv
Model 7:        CV = 1138.992+1560.6,   Name = model_l2_lr_0.01_trees_7K
Model 8:        CV = 1136.702+1569.7,   Name = model_l2_bopt_run1_index75
Model 9:        CV = 1136.977+1566.1,   Name = model_l2_bopt_run2_index92
Model 10:       CV = 1135.422+1549.2,   Name = model_xgb_3
Model 11:       CV = 1146.402+1565.6,   Name = test_xgb_A1-20161103T205502
Model 12:       CV = 1133.082+1546.8,   Name = xgb_Vladimir_base_score_1-20161103T230621
Model 13:       CV = 1142.349+1552.5,   Name = xgb-1117-attempt1-20161107T234751
Model 14:       CV = 1138.632+1564.7,   Name = ./keras_A5_1/oof_train.csv
Model 15:       CV = 1136.688+1567.8,   Name = ./keras_A5_2/oof_train.csv
Model 16:       CV = 1136.114+1558.9,   Name = ./keras_A5_11/oof_train.csv
Model 17:       CV = 1135.930+1563.4,   Name = lgbm_l2_loglossshift_extremin_1112-20161110T023325
Model 18:       CV = 1132.891+1557.3,   Name = lgbm_l2_loglossshift_extremin_1112-lr_0.003-20161110T031110
Model 19:       CV = 1133.808+1549.7,   Name = lgbm_l2_loglossshift500_extremin_1112-lr_0.01-20161113T203540
Model 20:       CV = 1146.014+1587.9,   Name = ./keras_A6_1/oof_train.csv
Model 21:       CV = 1146.831+1598.9,   Name = ./keras_A6_2/oof_train.csv
Model 22:       CV = 1138.803+1570.2,   Name = ./keras_A7_1/oof_train.csv
Model 23:       CV = 1140.436+1550.8,   Name = ./keras_A8_1/oof_train.csv
Model 24:       CV = 1131.660+1547.6,   Name = xgb-1109-attempt_1-20161114T223803
Model 25:       CV = 1127.974+1552.0,   Name = xgb-1109-attempt_2-20161115T231535
Model 26:       CV = 1133.127+1549.4,   Name = xgb-1109-attempt_3-20161116T091620
Model 27:       CV = 1219.379+1626.8,   Name = skl_Ridge-quad-100020161117T103454
Model 28:       CV = 1216.512+1629.9,   Name = skl_Ridge-quad-100020161117T104911
Model 29:       CV = 1216.830+1647.6,   Name = skl_Ridge-quad-100020161117T110254
Model 30:       CV = 1220.054+1627.3,   Name = skl_Ridge-quad-100020161117T111651
Model 31:       CV = 1217.454+1627.0,   Name = skl_Ridge-quad-100020161117T113232
Model 32:       CV = 1216.311+1705.7,   Name = skl_Ridge-quad-100020161117T114551
Model 33:       CV = 1217.437+1625.2,   Name = skl_Ridge-quad-100020161117T120118
Model 34:       CV = 1221.585+1634.2,   Name = skl_Ridge-quad-100020161117T121525
Model 35:       CV = 1220.311+1654.0,   Name = skl_Ridge-quad-100020161117T123053
Model 36:       CV = 1226.223+1728.5,   Name = skl_Ridge-quad-100020161117T124430
Model 37:       CV = 1136.613+1554.2,   Name = xgb-1109-attempt_4-base_score_7.7-20161119T202505
Model 38:       CV = 1134.472+1529.0,   Name = xgb-1109-attempt_5-preprocessor_pow_0.25_shift_1-20161119T225103
Model 39:       CV = 1134.353+1533.7,   Name = xgb-1109-attempt_5-preprocessor_pow_0.2_shift_10-20161120T055959
Model 40:       CV = 1131.255+1532.5,   Name = xgb-1109-attempt_8-obj_reglinear-preprocessor_pow_0.2_shift_10-20161120T225122
Model 41:       CV = 1140.101+1558.0,   Name = xgb-1109-attempt_9-obj_reglinear-preprocessor_log_shift_200-20161121T092757
Model 42:       CV = 1129.313+1545.3,   Name = xgb-mrooijer_1130-attempt_1-obj_fair1.5-preprocessor_pow_0.25_shift_1-base_score-8.4738505029380491-20161121T215220
Model 43:       CV = 1124.922+1540.4,   Name = xgb-mrooijer_1130-attempt_2-obj_fair1.5-preprocessor_pow_0.25_shift_1-base_score-8.4738505029380491-20161121T232803
Model 44:       CV = 1129.411+1551.2,   Name = xgb-mrooijer_1130-attempt_5-obj_fair0.7-preprocessor_pow_0.25_shift_1-base_score-8.4738505029380491-20161122T214600
Model 45:       CV = 1128.918+1541.5,   Name = xgb-mrooijer-lin-attempt_1-obj_fair1.5-preprocessor_pow_0.25_shift_1-base_score-8.4738505029380491-20161123T001744
Model 46:       CV = 1124.523+1536.9,   Name = xgb-mrooijer-lin-attempt_2-obj_fair1.5-preprocessor_pow_0.25_shift_1-base_score-8.4738505029380491-20161123T013341
Model 47:       CV = 1138.821+1565.0,   Name = ./keras_A9_1/oof_train.csv
Model 48:       CV = 1135.291+1531.5,   Name = ./keras_A9_2/oof_train.csv
Model 49:       CV = 1132.961+1530.5,   Name = ./keras_A9_4/oof_train.csv
(188318, 215),(125546, 215)

        model = Sequential()
        model.add(Dense(100, input_dim = l1_x_train.shape[1], init = 'he_normal'))
        model.add(BatchNormalization())
        model.add(PReLU())
        model.add(Dropout(0.3))
        model.add(Dense(1, init = 'he_normal'))
        model.compile(loss = 'mae', optimizer = 'adam')
        return(model)


# AM_2
LB = 1102.487
CV = 1120.753

Weighted Arithematic Mean
        (w/100*res_xgb_mrooijer_lin_A2['oof_test'] + res_keras_2['oof_test']*(1-w/100))**(1+p/10000)
        w = 65
        p = 7


# keras_A9_5
LB = 1115.94
CV = 1132.96 + std 8.3

**with log_shift_200 and base_score mean+1.0**

batch-size = 256
patience = 5 (with ModelCheckpoint)
nfolds = 5
nbags = 5
nepochs = 200

Dim train (188318, 1205)
Dim test (125546, 1205)

    model = Sequential()
    model.add(Dense(400, input_dim = xtrain.shape[1], init = 'he_normal'))
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
    model.add(Lambda(lambda x: x + 7.79 + 1.0))
    model.add(Lambda(lambda x: np.exp(x)))
    model.add(Lambda(lambda x: x-200))
    model.compile(loss = 'mae', optimizer = 'adam')
 

# keras_10_A1
LB = 1115.34
CV = 1129.47

**New Architecture : with more neurons in each hidden layer**
**Going back to adadelta as optimizer (last used in keras_2 ?)**

500-BN-PReLu-0.4
300-BN-PReLu-0.2
100-BN-PReLu-0.2
1
exp(p+7.79)-200

10 nfold
10 nbags
patience 5 (with modelcheckpoint)
batch size 128

X + lin_feats + diff_cont1_9

Updated order of loops for folds & bags: Now, Outer loop is bags, inner loop is folds


# xgb-mrooijer-lin-attempt_3-obj_fair1.5-preprocessor_pow_0.25_shift_1-base_score-8.4738505029380491-20161203T211419
LB = 1112.04
CV = 1132.82

**Lower max_depth to 4 from 12**
**Allow more colsample by tree**

x_train.shape = (188318, 740)

{'base_score': 8.4738505029380491,
 'booster': 'gbtree',
 **'colsample_bytree': 0.8,**
 'early_stopping_rounds': 100,
 'label_processor_function_name': 'pow_0.25_shift_1',
 'learning_rate': 0.1,
 **'max_depth': 4,**
 'min_child_weight': 100,
 'nrounds': 3500,
 'seed': 35791,
 'silent': 1,
 'subsample': 0.7,
 'verbose_eval': 10}

# xgb-mrooijer-lin-attempt_4-obj_fair1.5-preprocessor_pow_0.25_shift_1-base_score-8.4738505029380491-20161204T123341
LB = 1105.61
CV = 1123.78

**attempt_2 with larger colsample_bytree**

{'base_score': 8.4738505029380491,
 'booster': 'gbtree',
 **'colsample_bytree': 0.15,**
 'early_stopping_rounds': 100,
 'label_processor_function_name': 'pow_0.25_shift_1',
 'learning_rate': 0.01,
 'max_depth': 12,
 'min_child_weight': 100,
 'nrounds': 3500,
 'seed': 35791,
 'subsample': 0.7,
 'verbose_eval': 10}
 
# xgb-mrooijer-lin-attempt_5-obj_fair1.5-preprocessor_pow_0.25_shift_1-base_score-8.4738505029380491-20161205T101649
CV = 1128.84

**attempt_2 + ADD_DIFF and SUMS for all lin_feats**

x_train.shape = (188318, 921)

{'base_score': 8.4738505029380491,
 'booster': 'gbtree',
 'colsample_bytree': 0.08,
 'early_stopping_rounds': 100,
 'label_processor_function_name': 'pow_0.25_shift_1',
 'learning_rate': 0.01,
 'max_depth': 12,
 'min_child_weight': 100,
 'nrounds': 3500,
 'seed': 35791,
 'subsample': 0.7,
 'verbose_eval': 10}

# Thirteenth Keras Stacking
LB = 1101.12
CV = 1118.87 + std-10-fold 14.3

58 models
223 features

all features + lin_features + top-cats as dummies

nfolds = 10
**nbags = 2**
**nepochs = 30 (patience 5)**
nbags_per_fold = 1
Run time:  1h2m
Model 0:        CV = 1145.929+1581.4,   Name = model_fair_c_2_w_100_lr_0.002_trees_20K
Model 1:        CV = 1269.171+1569.8,   Name = model_et
Model 2:        CV = 1223.941+1693.0,   Name = skl_rf-20161108T012715
Model 3:        CV = 1138.184+1546.8,   Name = keras_1/oof_train_keras_400_0.4_200_0.2_nbags_4_nepochs_55_nfolds_5.csv
Model 4:        CV = 1133.942+1554.8,   Name = keras_2/oof_train.csv
Model 5:        CV = 1135.619+1564.4,   Name = keras_3/oof_train.csv
Model 6:        CV = 1140.481+1563.9,   Name = keras_A4/oof_train.csv
Model 7:        CV = 1138.992+1560.6,   Name = model_l2_lr_0.01_trees_7K
Model 8:        CV = 1136.702+1569.7,   Name = model_l2_bopt_run1_index75
Model 9:        CV = 1136.977+1566.1,   Name = model_l2_bopt_run2_index92
Model 10:       CV = 1135.422+1549.2,   Name = model_xgb_3
Model 11:       CV = 1146.402+1565.6,   Name = test_xgb_A1-20161103T205502
Model 12:       CV = 1133.082+1546.8,   Name = xgb_Vladimir_base_score_1-20161103T230621
Model 13:       CV = 1142.349+1552.5,   Name = xgb-1117-attempt1-20161107T234751
Model 14:       CV = 1138.632+1564.7,   Name = ./keras_A5_1/oof_train.csv
Model 15:       CV = 1136.688+1567.8,   Name = ./keras_A5_2/oof_train.csv
Model 16:       CV = 1136.114+1558.9,   Name = ./keras_A5_11/oof_train.csv
Model 17:       CV = 1135.930+1563.4,   Name = lgbm_l2_loglossshift_extremin_1112-20161110T023325
Model 18:       CV = 1132.891+1557.3,   Name = lgbm_l2_loglossshift_extremin_1112-lr_0.003-20161110T031110
Model 19:       CV = 1133.808+1549.7,   Name = lgbm_l2_loglossshift500_extremin_1112-lr_0.01-20161113T203540
Model 20:       CV = 1146.014+1587.9,   Name = ./keras_A6_1/oof_train.csv
Model 21:       CV = 1146.831+1598.9,   Name = ./keras_A6_2/oof_train.csv
Model 22:       CV = 1138.803+1570.2,   Name = ./keras_A7_1/oof_train.csv
Model 23:       CV = 1140.436+1550.8,   Name = ./keras_A8_1/oof_train.csv
Model 24:       CV = 1131.660+1547.6,   Name = xgb-1109-attempt_1-20161114T223803
Model 25:       CV = 1127.974+1552.0,   Name = xgb-1109-attempt_2-20161115T231535
Model 26:       CV = 1133.127+1549.4,   Name = xgb-1109-attempt_3-20161116T091620
Model 27:       CV = 1219.379+1626.8,   Name = skl_Ridge-quad-100020161117T103454
Model 28:       CV = 1216.512+1629.9,   Name = skl_Ridge-quad-100020161117T104911
Model 29:       CV = 1216.830+1647.6,   Name = skl_Ridge-quad-100020161117T110254
Model 30:       CV = 1220.054+1627.3,   Name = skl_Ridge-quad-100020161117T111651
Model 31:       CV = 1217.454+1627.0,   Name = skl_Ridge-quad-100020161117T113232
Model 32:       CV = 1216.311+1705.7,   Name = skl_Ridge-quad-100020161117T114551
Model 33:       CV = 1217.437+1625.2,   Name = skl_Ridge-quad-100020161117T120118
Model 34:       CV = 1221.585+1634.2,   Name = skl_Ridge-quad-100020161117T121525
Model 35:       CV = 1220.311+1654.0,   Name = skl_Ridge-quad-100020161117T123053
Model 36:       CV = 1226.223+1728.5,   Name = skl_Ridge-quad-100020161117T124430
Model 37:       CV = 1136.613+1554.2,   Name = xgb-1109-attempt_4-base_score_7.7-20161119T202505
Model 38:       CV = 1134.472+1529.0,   Name = xgb-1109-attempt_5-preprocessor_pow_0.25_shift_1-20161119T225103
Model 39:       CV = 1134.353+1533.7,   Name = xgb-1109-attempt_5-preprocessor_pow_0.2_shift_10-20161120T055959
Model 40:       CV = 1131.255+1532.5,   Name = xgb-1109-attempt_8-obj_reglinear-preprocessor_pow_0.2_shift_10-20161120T225122
Model 41:       CV = 1140.101+1558.0,   Name = xgb-1109-attempt_9-obj_reglinear-preprocessor_log_shift_200-20161121T092757
Model 42:       CV = 1129.313+1545.3,   Name = xgb-mrooijer_1130-attempt_1-obj_fair1.5-preprocessor_pow_0.25_shift_1-base_score-8.4738505029380491-20161121T215220
Model 43:       CV = 1124.922+1540.4,   Name = xgb-mrooijer_1130-attempt_2-obj_fair1.5-preprocessor_pow_0.25_shift_1-base_score-8.4738505029380491-20161121T232803
Model 44:       CV = 1129.411+1551.2,   Name = xgb-mrooijer_1130-attempt_5-obj_fair0.7-preprocessor_pow_0.25_shift_1-base_score-8.4738505029380491-20161122T214600
Model 45:       CV = 1128.918+1541.5,   Name = xgb-mrooijer-lin-attempt_1-obj_fair1.5-preprocessor_pow_0.25_shift_1-base_score-8.4738505029380491-20161123T001744
Model 46:       CV = 1124.523+1536.9,   Name = xgb-mrooijer-lin-attempt_2-obj_fair1.5-preprocessor_pow_0.25_shift_1-base_score-8.4738505029380491-20161123T013341
Model 47:       CV = 1138.821+1565.0,   Name = ./keras_A9_1/oof_train.csv
Model 48:       CV = 1135.291+1531.5,   Name = ./keras_A9_2/oof_train.csv
Model 49:       CV = 1132.961+1530.5,   Name = ./keras_A9_4/oof_train.csv
Model 50:       CV = 1132.691+1535.9,   Name = ./keras_A9_5/oof_train.csv
Model 51:       CV = 1132.487+1541.8,   Name = ./keras_A9_5_adadelta/oof_train.csv
Model 52:       CV = 1131.846+1542.0,   Name = ./keras_A9_6_nfold10_nbags3_batch128/oof_train.csv
Model 53:       CV = 1129.471+1533.7,   Name = ./keras_10_A1/oof_train.csv
Model 54:       CV = 1132.818+1534.6,   Name = xgb-mrooijer-lin-attempt_3-obj_fair1.5-preprocessor_pow_0.25_shift_1-base_score-8.4738505029380491-20161203T211419
Model 55:       CV = 1123.784+1533.7,   Name = xgb-mrooijer-lin-attempt_4-obj_fair1.5-preprocessor_pow_0.25_shift_1-base_score-8.4738505029380491-20161204T123341
Model 56:       CV = 1128.843+1534.9,   Name = xgb-mrooijer-lin-attempt_5-obj_fair1.5-preprocessor_pow_0.25_shift_1-base_score-8.4738505029380491-20161205T101649
AM_2 included
['lin_cont1', 'lin_cont2', 'lin_cont3', 'lin_cont4', 'lin_cont5', 'lin_cont6', 'lin_cont7', 'lin_cont8', 'lin_cont9', 'lin_cont10', 'lin_cont11', 'lin_cont12', 'lin_cont13', 'lin_cont14']
(188318, 223),(125546, 223)


# Thirteen-B Keras Stacking
LB = 1101.23
CV = 1118.63 + 7.8

nfolds = 10
**nbags = 8**
nepochs = 30 (patience 5)
**nbags_per_fold = 2**


# Fourteen Keras Stacking
LB = 1101.20
CV = 1119.62 + std-10-fold 5-7

58 models
223 features

**StratifiedKFold with 10 percentiles**
all features + lin_features + top-cats as dummies

nfolds = 10
nbags = 2
nepochs = 30 (patience 5)
nbags_per_fold = 1


# xgb-mrooijer-lin-attempt_6-obj_fair1.5-preprocessor_pow_0.25_shift_1-base_score-8.4738505029380491-20161207T230841
LB = 1104.71
CV = 1124.36


**attempt_2 + ali_cat from data.sort_cat.feat_comb.ali_cont**
*lin = ADD_DIFF_OLD*


# xgb-mrooijer-lin-attempt_7-obj_fair1.5-preprocessor_pow_0.25_shift_1-base_score-8.4738505029380491-20161208T120746
LB = 1104.77
CV = 1124.54

**attempt_6 + max_depth increased from 12 to 13 + gamma 1**

{'base_score': 8.4738505029380491,
 'booster': 'gbtree',
 'colsample_bytree': 0.08,
 'early_stopping_rounds': 100,
 **'gamma': 1,**
 'label_processor_function_name': 'pow_0.25_shift_1',
 'learning_rate': 0.01,
 **'max_depth': 13,**
 'min_child_weight': 100,
 'nrounds': 3500,
 'seed': 35791,
 'subsample': 0.7,
 'verbose_eval': 10}
 
# Fourteen-B Keras Stacking
Fourteen Keras Stacking + xgb-mrooijer-lin-attempt_6-obj_fair1.5-preprocessor_pow_0.25_shift_1-base_score-8.4738505029380491-20161207T230841
LB = 1101.39
CV = 1119.286

# keras_classify_10_A1

CV = 1203.49
NN to predict y_prec class out of 10


# keras_11_BN_deep_arch_A1
CV = 1141.6

**BN after PRelu**
**512-256-128-64-1 neurons**


# Fourteen-C Keras Stacking
LB = 1101.26
CV = 1120.33

Fourteen Keras Stacking + xgb-mrooijer-lin-attempt_6-obj_fair1.5-preprocessor_pow_0.25_shift_1-base_score-8.4738505029380491-20161207T230841
+ keras_classify_10_A1 + keras_11_BN_deep_arch_A1


# WeightedSum attempt-1 Stacking
LB = 1102.77
CV = 1118.66
Model 0:        CV = 1145.929+1581.4,   Name = model_fair_c_2_w_100_lr_0.002_trees_20K
Model 1:        CV = 1269.171+1569.8,   Name = model_et
Model 2:        CV = 1223.941+1693.0,   Name = skl_rf-20161108T012715
Model 3:        CV = 1138.184+1546.8,   Name = keras_1/oof_train_keras_400_0.4_200_0.2_nbags_4_nepochs_55_nfolds_5.csv
Model 4:        CV = 1133.942+1554.8,   Name = keras_2/oof_train.csv
Model 5:        CV = 1135.619+1564.4,   Name = keras_3/oof_train.csv
Model 6:        CV = 1140.481+1563.9,   Name = keras_A4/oof_train.csv
Model 7:        CV = 1138.992+1560.6,   Name = model_l2_lr_0.01_trees_7K
Model 8:        CV = 1136.702+1569.7,   Name = model_l2_bopt_run1_index75
Model 9:        CV = 1136.977+1566.1,   Name = model_l2_bopt_run2_index92
Model 10:       CV = 1135.422+1549.2,   Name = model_xgb_3
Model 11:       CV = 1146.402+1565.6,   Name = test_xgb_A1-20161103T205502
Model 12:       CV = 1133.082+1546.8,   Name = xgb_Vladimir_base_score_1-20161103T230621
Model 13:       CV = 1142.349+1552.5,   Name = xgb-1117-attempt1-20161107T234751
Model 14:       CV = 1138.632+1564.7,   Name = ./keras_A5_1/oof_train.csv
Model 15:       CV = 1136.688+1567.8,   Name = ./keras_A5_2/oof_train.csv
Model 16:       CV = 1136.114+1558.9,   Name = ./keras_A5_11/oof_train.csv
Model 17:       CV = 1135.930+1563.4,   Name = lgbm_l2_loglossshift_extremin_1112-20161110T023325
Model 18:       CV = 1132.891+1557.3,   Name = lgbm_l2_loglossshift_extremin_1112-lr_0.003-20161110T031110
Model 19:       CV = 1133.808+1549.7,   Name = lgbm_l2_loglossshift500_extremin_1112-lr_0.01-20161113T203540
Model 20:       CV = 1146.014+1587.9,   Name = ./keras_A6_1/oof_train.csv
Model 21:       CV = 1146.831+1598.9,   Name = ./keras_A6_2/oof_train.csv
Model 22:       CV = 1138.803+1570.2,   Name = ./keras_A7_1/oof_train.csv
Model 23:       CV = 1140.436+1550.8,   Name = ./keras_A8_1/oof_train.csv
Model 24:       CV = 1131.660+1547.6,   Name = xgb-1109-attempt_1-20161114T223803
Model 25:       CV = 1127.974+1552.0,   Name = xgb-1109-attempt_2-20161115T231535
Model 26:       CV = 1133.127+1549.4,   Name = xgb-1109-attempt_3-20161116T091620
Model 27:       CV = 1219.379+1626.8,   Name = skl_Ridge-quad-100020161117T103454
Model 28:       CV = 1216.512+1629.9,   Name = skl_Ridge-quad-100020161117T104911
Model 29:       CV = 1216.830+1647.6,   Name = skl_Ridge-quad-100020161117T110254
Model 30:       CV = 1220.054+1627.3,   Name = skl_Ridge-quad-100020161117T111651
Model 31:       CV = 1217.454+1627.0,   Name = skl_Ridge-quad-100020161117T113232
Model 32:       CV = 1216.311+1705.7,   Name = skl_Ridge-quad-100020161117T114551
Model 33:       CV = 1217.437+1625.2,   Name = skl_Ridge-quad-100020161117T120118
Model 34:       CV = 1221.585+1634.2,   Name = skl_Ridge-quad-100020161117T121525
Model 35:       CV = 1220.311+1654.0,   Name = skl_Ridge-quad-100020161117T123053
Model 36:       CV = 1226.223+1728.5,   Name = skl_Ridge-quad-100020161117T124430
Model 37:       CV = 1136.613+1554.2,   Name = xgb-1109-attempt_4-base_score_7.7-20161119T202505
Model 38:       CV = 1134.472+1529.0,   Name = xgb-1109-attempt_5-preprocessor_pow_0.25_shift_1-20161119T225103
Model 39:       CV = 1134.353+1533.7,   Name = xgb-1109-attempt_5-preprocessor_pow_0.2_shift_10-20161120T055959
Model 40:       CV = 1131.255+1532.5,   Name = xgb-1109-attempt_8-obj_reglinear-preprocessor_pow_0.2_shift_10-20161120T225122
Model 41:       CV = 1140.101+1558.0,   Name = xgb-1109-attempt_9-obj_reglinear-preprocessor_log_shift_200-20161121T092757
Model 42:       CV = 1129.313+1545.3,   Name = xgb-mrooijer_1130-attempt_1-obj_fair1.5-preprocessor_pow_0.25_shift_1-base_score-8.4738505029380491-20161121T215220
Model 43:       CV = 1124.922+1540.4,   Name = xgb-mrooijer_1130-attempt_2-obj_fair1.5-preprocessor_pow_0.25_shift_1-base_score-8.4738505029380491-20161121T232803
Model 44:       CV = 1129.411+1551.2,   Name = xgb-mrooijer_1130-attempt_5-obj_fair0.7-preprocessor_pow_0.25_shift_1-base_score-8.4738505029380491-20161122T214600
Model 45:       CV = 1128.918+1541.5,   Name = xgb-mrooijer-lin-attempt_1-obj_fair1.5-preprocessor_pow_0.25_shift_1-base_score-8.4738505029380491-20161123T001744
Model 46:       CV = 1124.523+1536.9,   Name = xgb-mrooijer-lin-attempt_2-obj_fair1.5-preprocessor_pow_0.25_shift_1-base_score-8.4738505029380491-20161123T013341
Model 47:       CV = 1138.821+1565.0,   Name = ./keras_A9_1/oof_train.csv
Model 48:       CV = 1135.291+1531.5,   Name = ./keras_A9_2/oof_train.csv
Model 49:       CV = 1132.961+1530.5,   Name = ./keras_A9_4/oof_train.csv
Model 50:       CV = 1132.691+1535.9,   Name = ./keras_A9_5/oof_train.csv
Model 51:       CV = 1132.487+1541.8,   Name = ./keras_A9_5_adadelta/oof_train.csv
Model 52:       CV = 1131.846+1542.0,   Name = ./keras_A9_6_nfold10_nbags3_batch128/oof_train.csv
Model 53:       CV = 1129.471+1533.7,   Name = ./keras_10_A1/oof_train.csv
Model 54:       CV = 1120.753+1532.6,   Name = weighted_mean_keras_2_xgb_moo_lin_A2_w_65_p_7
Model 55:       CV = 1132.818+1534.6,   Name = xgb-mrooijer-lin-attempt_3-obj_fair1.5-preprocessor_pow_0.25_shift_1-base_score-8.4738505029380491-20161203T211419
Model 56:       CV = 1123.784+1533.7,   Name = xgb-mrooijer-lin-attempt_4-obj_fair1.5-preprocessor_pow_0.25_shift_1-base_score-8.4738505029380491-20161204T123341
Model 57:       CV = 1128.843+1534.9,   Name = xgb-mrooijer-lin-attempt_5-obj_fair1.5-preprocessor_pow_0.25_shift_1-base_score-8.4738505029380491-20161205T101649
Model 58:       CV = 1124.366+1536.3,   Name = xgb-mrooijer-lin-attempt_6-obj_fair1.5-preprocessor_pow_0.25_shift_1-base_score-8.4738505029380491-20161207T230841
Model 59:       CV = 1203.491+1770.7,   Name = ./keras_classify_10_A1/oof_train.csv
Model 60:       CV = 1141.602+1590.2,   Name = ./keras_11_BN_deep_arch_A1/oof_train.csv
Model 61:       CV = 1169.670+1594.9,   Name = lgbm_fair_c_1.5_w_100-leaf_2000-lr_0.01-max_3500_trees-20161209T182238
(188318, 62),(125546, 62)
Ensamble Score: 1118.6573293728716
Best Weights: [  1.61136494e-13   6.13892200e-12   7.58829493e-12   1.40841645e-13
   1.90719185e-13   1.51534529e-13   1.33091173e-13   1.55567981e-13
   1.04651519e-13   1.13584093e-13   1.09255903e-13   1.33121276e-13
   1.61120869e-13   1.47027788e-13   2.50876181e-13   2.20636995e-13
   5.26483654e-02   7.12347524e-14   9.92101193e-14   9.21736609e-14
   1.19606785e-13   1.16288614e-12   1.91875166e-13   2.15993842e-13
   1.06586402e-02   9.55732847e-02   1.65088455e-13   5.86602746e-12
   5.06557197e-12   5.63575565e-12   5.99155476e-12   5.65534608e-12
   5.26187572e-12   5.43204021e-12   5.80686198e-12   4.96070362e-12
   7.20343964e-12   1.19039113e-13   1.54438503e-13   1.73913043e-13
   1.53989583e-03   6.19576893e-14   2.22764795e-06   1.48142675e-01
   2.22771499e-13   2.32596809e-13   2.48492406e-13   5.34061201e-14
   9.47345971e-02   2.26206574e-13   1.68695312e-02   1.43978465e-02
   3.42305249e-02   1.64149784e-01   2.45675940e-13   3.35117458e-02
   2.39493544e-01   2.15108039e-13   9.40473380e-02   7.97232873e-13
   1.34372997e-13   1.00562564e-12]

        def mae_loss_func(weights):
            ''' scipy minimize will pass the weights as a numpy array '''
            final_prediction = np.sum(weights * l1_x_train, axis=1)
            return mean_absolute_error(y_train, final_prediction)


        starting_values = np.random.uniform(size=l1_x_train.shape[1])

        cons = ({'type':'eq','fun':lambda w: 1-sum(w)})
        bounds = [(0,1)]*l1_x_train.shape[1]

        res = minimize(mae_loss_func, 
                   starting_values, 
                   method = 'SLSQP', 
                   bounds = bounds, 
                   constraints = cons,
                   options={'maxiter': 100000})

        best_score = res['fun']
        weights = res['x']
        
# Fifteen Keras Stacking with ModelCheckpoint
LB = 1101.28
CV = 1118.58

62 models, no other data

{'es_patience': 10,
 'nbags': 1,
 'nbags_per_fold': 1,
 'nepochs': 60,
 'nfolds': 5,
 'train_batch_size': 128}
 
# Fifteen-B Keras Stacking
LB = 1101.27
CV = 1118.43

{'es_patience': 10,
 'nbags': 15,
 'nbags_per_fold': 1,
 'nepochs': 60,
 'nfolds': 5,
 'train_batch_size': 128}
 
 
# xgb-mrooijer-lin-attempt_8-obj_fair1.5-preprocessor_pow_0.32_shift_1-base_score-13.60279000422951-20161210T172004
CV = 1128.799

attempt_1 + pow_0.32_shift_1

{'base_score': 13.60279000422951,
 'booster': 'gbtree',
 'colsample_bytree': 0.08,
 'early_stopping_rounds': 100,
 **'label_processor_function_name': 'pow_0.32_shift_1',**
 'learning_rate': 0.05,
 'max_depth': 12,
 'min_child_weight': 100,
 'nrounds': 700,
 'seed': 35791,
 'silent': 1,
 'subsample': 0.7,
 'verbose_eval': 10}

# attempt_9
CV = 1128.72
attempt_1 + fair 0.7

# attempt_10
CV = 1128.24
attempt_1 + Stratified folds
 
 
# xgb-mrooijer-lin-attempt_11-obj_fair1.5-preprocessor_pow_0.25_shift_1-base_score-8.4738505029380491-20161211T011318

LB = 1104.90
CV = 1125.099
(751.97, 2617.59) 

attempt_2 + Stratified folds


{'base_score': 8.473850502938049,
 'booster': 'gbtree',
 'colsample_bytree': 0.08,
 'early_stopping_rounds': 100,
 'label_processor_function_name': 'pow_0.25_shift_1',
 'learning_rate': 0.01,
 'max_depth': 12,
 'min_child_weight': 100,
 'nrounds': 4000,
 'seed': 35791,
 'silent': 1,
 'subsample': 0.7,
 'verbose_eval': 10}
 

# Sixteen Keras Stacking
LB = 1100.71
CV = 1118.17
(751.76, 2583.77)

Keras.stacker-20161211T130940.json
{'USE_STRATIFIED': True,
 'es_patience': 10,
 'model': 'class_name: Sequential\nconfig:\n- class_name: Dense\n  config:\n    W_constraint: null\n    W_regularizer: null\n    activation: linear\n    activity_regularizer: null\n    b_constraint: null\n    b_regularizer: null\n    batch_input_shape: !!python/tuple [null, 95]\n    bias: true\n    init: he_normal\n    input_dim: 95\n    input_dtype: float32\n    name: dense_9\n    output_dim: 100\n    trainable: true\n- class_name: BatchNormalization\n  config: {axis: -1, beta_regularizer: null, epsilon: 1.0e-05, gamma_regularizer: null,\n    mode: 0, momentum: 0.99, name: batchnormalization_5, trainable: true}\n- class_name: PReLU\n  config: {init: zero, name: prelu_5, trainable: true}\n- class_name: Dropout\n  config: {name: dropout_5, p: 0.3, trainable: true}\n- class_name: Dense\n  config: {W_constraint: null, W_regularizer: null, activation: linear, activity_regularizer: null,\n    b_constraint: null, b_regularizer: null, bias: true, init: he_normal, input_dim: null,\n    name: dense_10, output_dim: 1, trainable: true}\nkeras_version: 1.1.0\n',
 'nbags': 1,
 'nbags_per_fold': 1,
 'nepochs': 60,
 'nfolds': 5,
 'train_batch_size': 128}

~25min

# Sixteen-B Keras Stacking

CV = 1117.96
(753.9, 2574.15)

Keras.stacker-20161211T152048.json
{'USE_STRATIFIED': True,
 'es_patience': 10,
 'nbags': 3,
 'nbags_per_fold': 1,
 'nepochs': 60,
 'nfolds': 5,
 'train_batch_size': 128}


# Sixteen-C Keras Stacking
LB = 1100.72
CV = 1117.89
(753.0, 2577.35)

Keras.stacker-20161211T185744.json

{'USE_STRATIFIED': True,
 'es_patience': 10,
 **'nbags': 6,**
 'nbags_per_fold': 1,
 'nepochs': 60,
 'nfolds': 5,
 'train_batch_size': 128}

~ 3h15m


# keras_12_mae_on_pow_0.32_shift_1_Stratified
CV = 1139.32

nfolds = 5
nbags = 1
es_patience = 10
batch_szie = 128
prep_name = power_0.32_shift_1
optimizer = adadelta
USE_STRATIFIED = True


400-BN-PReLU-0.4
200-BN-PReLU-0.2
50-BN-PReLU-0.2
1

class_name: Sequential\nconfig:\n- class_name: Dense\n  config:\n    W_constraint: null\n    W_regularizer: null\n    activation: linear\n    activity_regularizer: null\n    b_constraint: null\n    b_regularizer: null\n    batch_input_shape: !!python/tuple [null, 1204]\n    bias: true\n    init: he_normal\n    input_dim: 1204\n    input_dtype: float32\n    name: dense_17\n    output_dim: 400\n    trainable: true\n- class_name: BatchNormalization\n  config: {axis: -1, beta_regularizer: null, epsilon: 1.0e-05, gamma_regularizer: null,\n    mode: 0, momentum: 0.99, name: batchnormalization_13, trainable: true}\n- class_name: PReLU\n  config: {init: zero, name: prelu_13, trainable: true}\n- class_name: Dropout\n  config: {name: dropout_13, p: 0.4, trainable: true}\n- class_name: Dense\n  config: {W_constraint: null, W_regularizer: null, activation: linear, activity_regularizer: null,\n    b_constraint: null, b_regularizer: null, bias: true, init: he_normal, input_dim: null,\n    name: dense_18, output_dim: 200, trainable: true}\n- class_name: BatchNormalization\n  config: {axis: -1, beta_regularizer: null, epsilon: 1.0e-05, gamma_regularizer: null,\n    mode: 0, momentum: 0.99, name: batchnormalization_14, trainable: true}\n- class_name: PReLU\n  config: {init: zero, name: prelu_14, trainable: true}\n- class_name: Dropout\n  config: {name: dropout_14, p: 0.2, trainable: true}\n- class_name: Dense\n  config: {W_constraint: null, W_regularizer: null, activation: linear, activity_regularizer: null,\n    b_constraint: null, b_regularizer: null, bias: true, init: he_normal, input_dim: null,\n    name: dense_19, output_dim: 50, trainable: true}\n- class_name: BatchNormalization\n  config: {axis: -1, beta_regularizer: null, epsilon: 1.0e-05, gamma_regularizer: null,\n    mode: 0, momentum: 0.99, name: batchnormalization_15, trainable: true}\n- class_name: PReLU\n  config: {init: zero, name: prelu_15, trainable: true}\n- class_name: Dropout\n  config: {name: dropout_15, p: 0.2, trainable: true}\n- class_name: Dense\n  config: {W_constraint: null, W_regularizer: null, activation: linear, activity_regularizer: null,\n    b_constraint: null, b_regularizer: null, bias: true, init: he_normal, input_dim: null,\n    name: dense_20, output_dim: 1, trainable: true}\nkeras_version: 1.1.0\n'



# Sixteen-D Keras Stacking
LB = 1100.81
CV = 1117.84

**Sixteen-C + keras_12**
**3 bags**


# xgb-moo-lin-attempt_12
CV = 1218.07

**obj: fair 1.5 with X = X * log(y+9000)**
prep: pw_0.25_shift_1
eta = 0.05


# xgb-moo-lin-attempt_13
CV = 1127.12
(753.3 2622)

**pow_0.32_shift_1  + fair_0.7   +  stratifiedfolds**
**attempt_8         + attempt_9  +  attempt_10 with eta 0.01**

eta = 0.01


# Sixteen-E Keras Stacking
CV = 1117.786

**Sixteen-D + xgb-moo-lin-A{12,13}**
nbags = 3


# Seventeen Keras Stacking
LB = 1101.08
CV = 1117.61
(751.99, 2580)

**New Architecture (2 hidden layers) + 16-E**
100-BN-PReLU-0.3
-50-BN-PReLU-0.3
-1


(188318, 98)

{'USE_STRATIFIED': True,
 'es_patience': 10,
 'model': 'class_name: Sequential\nconfig:\n- class_name: Dense\n  config:\n    W_constraint: null\n  W_regularizer: null\n    activation: linear\n    activity_regularizer: null\n    b_constraint: null\n    b_regularizer: null\n    batch_input_shape: !!python/tuple [null, 98]\n    bias: true\n    init:he_normal\n    input_dim: 98\n    input_dtype: float32\n    name: dense_43\n    output_dim: 100\n    trainable: true\n- class_name: BatchNormalization\n  config: {axis: -1, beta_regularizer: null, epsilon: 1.0e-05, gamma_regularizer: null,\n    mode: 0, momentum: 0.99, name: batchnormalization_29, trainable: true}\n- class_name: PReLU\n  config: {init: zero, name: prelu_29, trainable: true}\n- class_name: Dropout\n  config: {name: dropout_29, p: 0.3, trainable: true}\n- class_name: Dense\n  config: {W_constraint: null, W_regularizer: null, activation: linear, activity_regularizer: null,\n    b_constraint: null, b_regularizer: null, bias: true, init: he_normal, input_dim: null,\n    name: dense_44, output_dim: 50, trainable: true}\n- class_name: BatchNormalization\n  config: {axis: -1, beta_regularizer: null, epsilon: 1.0e-05, gamma_regularizer: null,\n    mode: 0, momentum: 0.99, name: batchnormalization_30, trainable: true}\n- class_name: PReLU\n  config: {init: zero, name: prelu_30, trainable: true}\n-class_name: Dropout\n  config: {name: dropout_30, p: 0.3, trainable: true}\n- class_name: Dense\n  config: {W_constraint: null, W_regularizer: null, activation: linear, activity_regularizer: null,\n    b_constraint: null, b_regularizer: null, bias: true, init: he_normal, input_dim: null,\n    name: dense_45, output_dim: 1, trainable: true}\nkeras_version: 1.1.0\n',
 'nbags': 3,
 'nbags_per_fold': 1,
 'nepochs': 60,
 'nfolds': 5,
 'train_batch_size': 128}
 
 
# Weighted Average of Keras_16_E and Keras_17
LB = 1100.74
CV = 1117.3090

keras_16E * 45/100 + keras_17 * 55/100

keras_16E CV = 1117.786
keras_17 CV = 1117.61


----

res_16      1118.171  *751.766*  2583.771       1100.714        Selected
res_16B     1117.962   753.909  *2574.156*
res_16C     1117.890   753.021   2577.349       1100.724
res_16D     1117.841   752.428   2579.473       1100.817
res_16E     1117.787   753.275   2575.815
res_17     *1117.620*  752.000   2580.079       1101.081
W_16E_17    1117.309   752.198   2577.733       1100.740        Selected

----




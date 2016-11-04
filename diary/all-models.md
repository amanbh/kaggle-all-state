# Fifth Stacking (with XGB)
LB = 1114.27

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
Vladimir XGB-1114 params + base_score = 1 + higher eta
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





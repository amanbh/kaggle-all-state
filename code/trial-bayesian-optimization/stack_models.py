sys.exit()

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
    'learning_rate': 0.01,
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
    'metric_freq': 500,
    'early_stopping_round': 100,
}

# lg_l2 = LightgbmWrapper(seed=SEED, params=lightgbm_params_l2)
# lg_oof_train_l2, lg_oof_test_l2 = get_oof(lg_l2)
res_lg_l2 = load_results_from_json('model_l2_lr_0.01_trees_7K.json')
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

submission = load_submission()
submission.iloc[:, 1] = gbdt.predict(dtest)
submission.to_csv('xgstacker_starter.sub.csv', index=None)

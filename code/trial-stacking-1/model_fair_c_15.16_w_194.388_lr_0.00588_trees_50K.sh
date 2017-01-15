#!/bin/bash

FOLD=0
../../../LightGBM/lightgbm \
    config=model_fair_c_15.16_w_194.388_lr_0.00588_trees_50K.train.conf \
    data=../trial-bayesian-optimization/pre_split_5_folds/train_fold_`echo ${FOLD}`.csv \
    test_data=../trial-bayesian-optimization/pre_split_5_folds/cv_fold_`echo ${FOLD}`.csv \
    output_model=LightGBM_model_`echo ${FOLD}`.txt

../../../LightGBM/lightgbm \
    task=predict \
    input_model=LightGBM_model_`echo ${FOLD}`.txt \
    data=../trial-bayesian-optimization/pre_split_5_folds/cv_as_test_fold_`echo ${FOLD}`.csv \
    output_result=LightGBM_predict_result_oof_`echo ${FOLD}`.txt

../../../LightGBM/lightgbm \
    task=predict \
    input_model=LightGBM_model_`echo ${FOLD}`.txt \
    data=../trial-fair-lightgbm/lgbm_test.csv \
    output_result=LightGBM_predict_result_test_`echo ${FOLD}`.txt

FOLD=1
../../../LightGBM/lightgbm \
    config=model_fair_c_15.16_w_194.388_lr_0.00588_trees_50K.train.conf \
    data=../trial-bayesian-optimization/pre_split_5_folds/train_fold_`echo ${FOLD}`.csv \
    test_data=../trial-bayesian-optimization/pre_split_5_folds/cv_fold_`echo ${FOLD}`.csv \
    output_model=LightGBM_model_`echo ${FOLD}`.txt

../../../LightGBM/lightgbm \
    task=predict \
    input_model=LightGBM_model_`echo ${FOLD}`.txt \
    data=../trial-bayesian-optimization/pre_split_5_folds/cv_as_test_fold_`echo ${FOLD}`.csv \
    output_result=LightGBM_predict_result_oof_`echo ${FOLD}`.txt

../../../LightGBM/lightgbm \
    task=predict \
    input_model=LightGBM_model_`echo ${FOLD}`.txt \
    data=../trial-fair-lightgbm/lgbm_test.csv \
    output_result=LightGBM_predict_result_test_`echo ${FOLD}`.txt

FOLD=2
../../../LightGBM/lightgbm \
    config=model_fair_c_15.16_w_194.388_lr_0.00588_trees_50K.train.conf \
    data=../trial-bayesian-optimization/pre_split_5_folds/train_fold_`echo ${FOLD}`.csv \
    test_data=../trial-bayesian-optimization/pre_split_5_folds/cv_fold_`echo ${FOLD}`.csv \
    output_model=LightGBM_model_`echo ${FOLD}`.txt

../../../LightGBM/lightgbm \
    task=predict \
    input_model=LightGBM_model_`echo ${FOLD}`.txt \
    data=../trial-bayesian-optimization/pre_split_5_folds/cv_as_test_fold_`echo ${FOLD}`.csv \
    output_result=LightGBM_predict_result_oof_`echo ${FOLD}`.txt

../../../LightGBM/lightgbm \
    task=predict \
    input_model=LightGBM_model_`echo ${FOLD}`.txt \
    data=../trial-fair-lightgbm/lgbm_test.csv \
    output_result=LightGBM_predict_result_test_`echo ${FOLD}`.txt

FOLD=3
../../../LightGBM/lightgbm \
    config=model_fair_c_15.16_w_194.388_lr_0.00588_trees_50K.train.conf \
    data=../trial-bayesian-optimization/pre_split_5_folds/train_fold_`echo ${FOLD}`.csv \
    test_data=../trial-bayesian-optimization/pre_split_5_folds/cv_fold_`echo ${FOLD}`.csv \
    output_model=LightGBM_model_`echo ${FOLD}`.txt

../../../LightGBM/lightgbm \
    task=predict \
    input_model=LightGBM_model_`echo ${FOLD}`.txt \
    data=../trial-bayesian-optimization/pre_split_5_folds/cv_as_test_fold_`echo ${FOLD}`.csv \
    output_result=LightGBM_predict_result_oof_`echo ${FOLD}`.txt

../../../LightGBM/lightgbm \
    task=predict \
    input_model=LightGBM_model_`echo ${FOLD}`.txt \
    data=../trial-fair-lightgbm/lgbm_test.csv \
    output_result=LightGBM_predict_result_test_`echo ${FOLD}`.txt

FOLD=4
../../../LightGBM/lightgbm \
    config=model_fair_c_15.16_w_194.388_lr_0.00588_trees_50K.train.conf \
    data=../trial-bayesian-optimization/pre_split_5_folds/train_fold_`echo ${FOLD}`.csv \
    test_data=../trial-bayesian-optimization/pre_split_5_folds/cv_fold_`echo ${FOLD}`.csv \
    output_model=LightGBM_model_`echo ${FOLD}`.txt

../../../LightGBM/lightgbm \
    task=predict \
    input_model=LightGBM_model_`echo ${FOLD}`.txt \
    data=../trial-bayesian-optimization/pre_split_5_folds/cv_as_test_fold_`echo ${FOLD}`.csv \
    output_result=LightGBM_predict_result_oof_`echo ${FOLD}`.txt

../../../LightGBM/lightgbm \
    task=predict \
    input_model=LightGBM_model_`echo ${FOLD}`.txt \
    data=../trial-fair-lightgbm/lgbm_test.csv \
    output_result=LightGBM_predict_result_test_`echo ${FOLD}`.txt


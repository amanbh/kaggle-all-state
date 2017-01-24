This repository contains the code used for the [Kaggle Allstate Claims Severity competition](https://www.kaggle.com/c/allstate-claims-severity).

## My Approach
The solution used in this code consists of stacking base multiple models with a small MLP model. 

### Base Models

For my base models I used:
 - [XGBoost](https://github.com/dmlc/xgboost)
 - [Keras](https://github.com/fchollet/keras) MLP
 - [LightGBM](https://github.com/Microsoft/LightGBM)
 - [scikit-learn](http://scikit-learn.org)
    * Ridge Regressor
    * RandomForest Regressor
    * ExtraTrees


Different models used different hyper-parameters, objective funtions, feature-interactions, target transformations to hande the MAE cost function as well as the (approximately) log-normal distribution of the target variable. I used 10-fold CV for validation with fixed seed.

The best single model was a xgb model with following hyper-parameters:

```python
{'base_score': 8.4738505029380491,
 'booster': 'gbtree',
 'colsample_bytree': 0.15,
 'early_stopping_rounds': 100, 
 'label_processor_function_name': 'pow_0.25_shift_1',
 'learning_rate': 0.01,
 'max_depth': 12,
 'min_child_weight': 100, 
 'nrounds': 3500,
 'subsample': 0.7,
 'verbose_eval': 10}
```
The features used for this model included the linearized features derived from the continuous-valued inputs. The objective function was fair with constant 1.5 on target variable transformed as
```
lambda x : pow(x, 0.25) + 1
```
The base score is obtained as mean(transformed-loss) + 1.

 
This model gave MAE = 1123.78 on the 10-fold CV, and MAE = 1105.61 on the public leaderboard.


### Stacking Model

For stacking, Xgboost did not work as well as Keras MLP. I used out-of-fold predictions from all base models to train Keras model.


## Lessons from this contest

- Don't spend a lot of time on  a single task for kaggle contests.
	* Hyper-parameter optimization, MLP architecture optimization without a GPU ;( was wasteful.
	* Could not spend as much time as I wanted on post-processing and feature/target transformations.
- Build and use classes for a sklearn-like fit/predict approach.
	* Eventually I built wrappers for predict oof/test on Xgboost, LightGBM, and scikit-learn models. 
- Don't be afraid to try crazy ideas.
	* Didn't try negative weights in blending, this worked for others.
	* Should have used more tranformations on categorical and numerical features that I could justify reasonably.


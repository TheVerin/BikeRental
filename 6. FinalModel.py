import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category = DeprecationWarning)

#Random Forest Regression
from sklearn.ensemble import RandomForestRegressor
RF = RandomForestRegressor(n_estimators = 200)
RF.fit(X, Y)

import xgboost as xg
XGB = xg.XGBRegressor()
XGB.fit(X, Y)

from sklearn.model_selection import GridSearchCV, StratifiedKFold
parameters = {'n_estimators':[ 50, 100, 200],
              'max_features':['sqrt', 'auto', 'log2'],
              'max_depth':[2, 4, 6, 10],
              'min_samples_split':[2,3,5],
              'min_samples_leaf':[1,3,5]}
cross_validation = StratifiedKFold(n_splits = 5)
Evaluation = GridSearchCV(RF, scoring = 'neg_mean_squared_log_error', param_grid = parameters, cv = cross_validation, verbose = 1)
Evaluation.fit(X, Y)
bests = Evaluation.best_params_
print('Best score: {}'.format(Evaluation.best_score_))
print('Best parameters: {}'.format(Evaluation.best_params_))

#Checking the prediction
RF_pred = RF.predict(Bike_test)
XGB_pred = XGB.predict(Bike_test)
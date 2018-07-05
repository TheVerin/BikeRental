import pandas as pd
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category = DeprecationWarning)

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 200)
regressor.fit(X_val, Y_val)


#Choosing the features to train
features = pd.DataFrame()
features['feature'] = X.columns
features['importance'] = regressor.feature_importances_
features.sort_values(by = 'importance', ascending = True, inplace = True)
features.set_index('feature', inplace = True)
features.plot(kind = 'barh')

import xgboost as xg
XGB = xg.XGBRegressor()
XGB.fit(X_val, Y_val)

#Grid search -> you have to finished that
from sklearn.model_selection import GridSearchCV, StratifiedKFold
parameters = {'n_estimators':[10, 50, 100],
              'max_features':['sqrt', 'auto', 'log2'],
              'max_depth':[2, 4, 6, 8, 10],
              'min_samples_split':[2,3,5,8],
              'min_samples_leaf':[1,3,5,8]}
cross_validation = StratifiedKFold(n_splits = 5)
grid_search = GridSearchCV(regressor, scoring = 'neg_mean_squared_log_error', param_grid = parameters, cv = cross_validation, verbose = 1)
grid_search.fit(X_val, Y_val)
bests = grid_search.best_params_
print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))


#Checking the prediction
Bike_pred = XGB.predict(X_train)

assert len(Y_train) == len(Bike_pred)
terms_to_sum = [(math.log(Bike_pred[i] + 1) - math.log(Y_train[i] + 1)) ** 2.0 for i,pred in enumerate(Bike_pred)]
score  = (sum(terms_to_sum) * (1.0/len(Y_train))) ** 0.5
print(score)
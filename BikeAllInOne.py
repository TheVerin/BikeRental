#Bike _1

#Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category = DeprecationWarning)
from scipy import stats
import math


#First view into data
Bike = pd.read_csv('train.csv')
Bike_test = pd.read_csv('test.csv')

Bike.describe()
Bike_test.describe()
#we may see that there is no missing data :D

Bike.dtypes
Bike_test.dtypes

Bike.skew()
Bike_test.skew()

Bike.hist(bins = 50)
Bike.select_dtypes([float, int]).apply(stats.normaltest)


#Visualising the data
#Scatter plot of atemp/temp ratio
plt.scatter(x = Bike['temp'], y = Bike['atemp'])
plt.scatter(x = Bike_test['temp'], y = Bike_test['atemp'])
#It is quite linear connesction

#Corelation of season/weather in case of weather
sns.stripplot(x = 'season', y = 'count', hue = 'weather', data = Bike)
#Truly? I think that weather is not connected with number of rents at all :(

#Comparing number of casual rentiers to registered in case of season
Bike.groupby('season').agg('sum')[['registered', 'casual']].plot(kind = 'bar', stacked = True, colors = ['g', 'r'])
#There is much moore registered riders than casuals, and the biggest number of rentiers is for fall season (3)

#Comparing number of casual rentiers to registered in case of weather
Bike.groupby('weather').agg('sum')[['registered', 'casual']].plot(kind = 'bar', stacked = True, colors = ['g', 'r'])
#There is a big diffrence beetwen riders in case of weather

#Split datatime into year, month and hour column
year = set()
for datetime in Bike['datetime']:
    year.add(datetime.split('-')[0].strip())
print (year)
YearDictionary = {'2011':2011, '2012':2012}

month = set()
for datetime in Bike['datetime']:
    month.add(datetime.split('-')[1].split('-')[0].strip())
print (month)
MonthDictionary = {'01':1, '02':2, '03':3, '04':4, '05':5, '06':6, '07':7, '08':8, '09':9, '10':10, '11':11, '12':12}

hour = set()
for datetime in Bike['datetime']:
    hour.add(datetime.split(' ')[1].split(':')[0].strip())
print (hour)
HourDictionary = {'01':1, '02':2, '03':3, '04':4, '05':5, '06':6, '07':7, '08':8, '09':9, '10':10, '11':11, '12':12,
                  '13':13, '14':14, '15':15, '16':16, '17':17, '18':18, '19':19, '20':20, '21':21, '22':22, '23':23, '00':00}


#Making new columns
def Bike_Year():
    global Bike
    Bike['year'] = Bike['datetime'].map(lambda datetime:datetime.split('-')[0].strip())
    Bike['year'] = Bike.year.map(YearDictionary)
    return Bike
Bike = Bike_Year()

def Bike_Month():
    global Bike
    Bike['month'] = Bike['datetime'].map(lambda datetime:datetime.split('-')[1].split('-')[0].strip())
    Bike['month'] = Bike.month.map(MonthDictionary)
    return Bike
Bike = Bike_Month()

def Bike_Hour():
    global Bike
    Bike['hour'] = Bike['datetime'].map(lambda datetime:datetime.split(' ')[1].split(':')[0].strip())
    Bike['hour'] = Bike.hour.map(HourDictionary)
    return Bike
Bike = Bike_Hour()

Bike.drop('datetime', axis = 1, inplace = True)

#Comming back to visualisation :3
#Number of rentings in each -> hour, month, year
sns.stripplot(data = Bike, x = 'hour', y = 'count', hue = 'season')
sns.stripplot(data = Bike, x = 'hour', y = 'count', hue = 'weather')
sns.stripplot(data = Bike, x = 'hour', y = 'count', hue = 'holiday')
#What about new dictionary? The number for each hour will be replaced by the dominant of rentings in this hour

sns.stripplot(data = Bike, x = 'month', y = 'count', hue = 'season')
sns.stripplot(data = Bike, x = 'month', y = 'count', hue = 'weather')

sns.stripplot(data = Bike, x = 'year', y = 'count', hue = 'season')
sns.stripplot(data = Bike, x = 'year', y = 'count', hue = 'weather')

sns.stripplot(data = Bike, x = 'atemp', y = 'count')
sns.stripplot(data = Bike, x = 'temp', y = 'count')

sns.stripplot(data = Bike, x = 'humidity', y = 'count')
sns.stripplot(data = Bike, x = 'windspeed', y = 'count')

sns.boxplot(data = Bike, x = 'windspeed', y = 'count')
sns.boxplot(data = Bike, x = 'atemp', y = 'count')
sns.boxplot(data = Bike, x = 'humidity', y = 'count')

sns.stripplot(data = Bike, x = 'humidity', y = 'humidity')
sns.stripplot(data = Bike, x = 'atemp', y = 'atemp')
sns.stripplot(data = Bike, x = 'windspeed', y = 'windspeed')


X = Bike.iloc[:, [0,2,3,5,6,7,11,12,13]]
Y = Bike.iloc[:, 10].values



#getting dummies
def WeatherDummies():
    global X
    Weather_Dummies = pd.get_dummies(X['weather'], prefix = 'weather')
    X = pd.concat([X, Weather_Dummies], axis = 1)
    X.drop('weather', axis = 1, inplace = True)
    return X
X = WeatherDummies()
X.drop('weather_4', axis = 1, inplace = True)

def SeasonDummies():
    global X
    Season_Dummies = pd.get_dummies(X['season'], prefix = 'season')
    X = pd.concat([X, Season_Dummies], axis = 1)
    X.drop('season', axis = 1, inplace = True)
    return X
X = SeasonDummies()
X.drop('season_4', axis = 1, inplace = True)

#Scalling the variables
from sklearn.preprocessing import StandardScaler
X_sc = StandardScaler()
X = X_sc.fit_transform(X)

#Spliting dataset into train and validation set
from sklearn.cross_validation import train_test_split
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, train_size = 0.2, random_state = 0)

#Random Forest Regression
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


#Making new columns
def Bike_test_Year():
    global Bike_test
    Bike_test['year'] = Bike_test['datetime'].map(lambda datetime:datetime.split('-')[0].strip())
    Bike_test['year'] = Bike_test.year.map(YearDictionary)
    return Bike_test
Bike_test = Bike_test_Year()

def Bike_test_Month():
    global Bike_test
    Bike_test['month'] = Bike_test['datetime'].map(lambda datetime:datetime.split('-')[1].split('-')[0].strip())
    Bike_test['month'] = Bike_test.month.map(MonthDictionary)
    return Bike_test
Bike_test = Bike_test_Month()

def Bike_test_Hour():
    global Bike_test
    Bike_test['hour'] = Bike_test['datetime'].map(lambda datetime:datetime.split(' ')[1].split(':')[0].strip())
    Bike_test['hour'] = Bike_test.hour.map(HourDictionary)
    return Bike_test
Bike_test = Bike_test_Hour()

Bike_test.drop(['datetime', 'temp'], axis = 1, inplace = True)

def WeatherDummies_test():
    global Bike_test
    Weather_Dummies_test = pd.get_dummies(Bike_test['weather'], prefix = 'weather')
    Bike_test = pd.concat([Bike_test, Weather_Dummies_test], axis = 1)
    Bike_test.drop('weather', axis = 1, inplace = True)
    return Bike_test
Bike_test = WeatherDummies_test()
Bike_test.drop('weather_4', axis = 1, inplace = True)

def SeasonDummies_test():
    global Bike_test
    Season_Dummies_test = pd.get_dummies(Bike_test['season'], prefix = 'season')
    Bike_test = pd.concat([Bike_test, Season_Dummies_test], axis = 1)
    Bike_test.drop('season', axis = 1, inplace = True)
    return Bike_test
Bike_test = SeasonDummies_test()
Bike_test.drop('season_4', axis = 1, inplace = True)

Bike_test.drop(['holiday'], axis = 1, inplace = True)

Bike_test['windspeed'] = abs(Bike_test['windspeed'].max() - Bike_test['windspeed'])

from sklearn.preprocessing import StandardScaler
Bike_test_sc = StandardScaler()
Bike_test = Bike_test_sc.fit_transform(X)

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

#preparing the final folder
df_output = pd.DataFrame()
aux = pd.read_csv('test.csv')
df_output['datetime'] = aux['datetime']
df_output['count'] = XGB.predict(Bike_test)
df_output[['datetime','count']].to_csv('XGB.csv', index=False)
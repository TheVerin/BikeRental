import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category = DeprecationWarning)
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
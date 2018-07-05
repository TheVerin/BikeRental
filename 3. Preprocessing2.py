import pandas as pd
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category = DeprecationWarning)

#Splitting for dependent and independent values
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
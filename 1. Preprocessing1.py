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

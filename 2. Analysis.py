import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category = DeprecationWarning)

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
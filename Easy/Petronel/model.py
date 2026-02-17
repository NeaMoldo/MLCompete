#imports
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

import warnings

warnings.filterwarnings('ignore')

#train
train = pd.read_csv('./dataset_train.csv')

print(f'{train.head()}\n')
print(f'{train.info()}\n')
print(f'{train.describe()}\n')
print(f'{train.isnull().sum()}\n')

#test
test = pd.read_csv('./dataset_eval.csv')

#Req1
task1 = train.copy()

task1['Elapsed Time'] = task1['Elapsed Time']/3600
task1['speed'] = task1['Distance']/task1['Elapsed Time']

task1['datetime'] = pd.to_datetime(task1['Activity Date'], format='%b %d, %Y, %I:%M:%S %p')
task1['month'] = task1['datetime'].dt.strftime('%b')

months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

rows = []
for m in months:
    month_data = task1[task1['month'] == m]
    avg_speed = month_data['speed'].mean() if not month_data.empty else 0
    avg_speed = np.floor(avg_speed * 1e5) / 1e5
    rows.append({'subtaskID': 1, 'Answer1': m, 'Answer2': avg_speed})

req1 = pd.DataFrame(rows)

#Req2
req2 = test.copy()
req2['subtaskID'] = 2
req2['Answer1'] = req2['Activity ID']

X = train.copy()

le = LabelEncoder()
X['Label'] = le.fit_transform(X['Label'])
X['TimeDiff'] = X['Elapsed Time']-X['Moving Time']

lat1, lon1 = np.radians(X['Starting Latitude']), np.radians(X['Starting Longitude'])
lat2, lon2 = np.radians(X['Finish Latitude']), np.radians(X['Finish Longitude'])
dlat = lat2 - lat1
dlon = lon2 - lon1
a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
c = 2 * np.arcsin(np.sqrt(a))
X['start_end_distance'] = 6371 * c 

y = X['Label']
X = X.drop(labels=(['Activity ID','Activity Date','Label','Moving Time','Elapsed Time','Starting Latitude','Starting Longitude','Finish Latitude','Finish Longitude']),axis=1)

forest = RandomForestClassifier()
forest.fit(X,y)

req2['TimeDiff'] = req2['Elapsed Time'] - req2['Moving Time']

lat1, lon1 = np.radians(req2['Starting Latitude']), np.radians(req2['Starting Longitude'])
lat2, lon2 = np.radians(req2['Finish Latitude']), np.radians(req2['Finish Longitude'])
dlat = lat2 - lat1
dlon = lon2 - lon1
a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
c = 2 * np.arcsin(np.sqrt(a))
req2['start_end_distance'] = 6371 * c 

features = req2.drop(labels=(['subtaskID','Answer1','Activity ID','Activity Date','Moving Time','Elapsed Time','Starting Latitude','Starting Longitude','Finish Latitude','Finish Longitude']),axis=1)


req2['Answer2'] = le.inverse_transform(forest.predict(features))

req2 = req2[['subtaskID', 'Answer1', 'Answer2']]

#submission
submission = pd.concat((req1,req2))

submission.columns = ['subtaskID', 'Answer1', 'Answer2']

submission.to_csv('submission.csv',index=False)
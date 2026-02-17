#imports
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

#dataset
train = pd.read_csv('./train.csv')

print(f"{train.head()}\n")
print(f"{train.tail()}\n")
print(f"{train.info()}\n")
print(f"{train.describe()}\n")
print(f"{train.isnull().sum()}\n")

train.drop(labels=['ticket_price','comfort_class','SampleID'],axis=1,inplace=True)

train = pd.get_dummies(train, columns=['weather','weekday'],drop_first=True)

train['departure_time'] = pd.to_datetime(train['departure_time'], format='%H:%M')
minutes = train['departure_time'].dt.hour * 60 + train['departure_time'].dt.minute

train['dep_sin'] = np.sin(2 * np.pi * minutes / 1440)
train['dep_cos'] = np.cos(2 * np.pi * minutes / 1440)

train = train.drop(columns=['departure_time'])

X = train.drop(labels=('delay_minutes'),axis=1)
y = train['delay_minutes']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)

#model
forest = RandomForestRegressor()
forest.fit(X_train,y_train)

#eval
y_pred = forest.predict(X_test)
print(f"{mean_absolute_error(y_test,y_pred)}\n")

#submission
test = pd.read_csv('./test.csv')

sample_id = test['SampleID']

test.drop(labels=['ticket_price','comfort_class','SampleID'],axis=1,inplace=True)

test = pd.get_dummies(test, columns=['weather','weekday'],drop_first=True)

test['departure_time'] = pd.to_datetime(test['departure_time'], format='%H:%M')
minutes = test['departure_time'].dt.hour * 60 + test['departure_time'].dt.minute

test['dep_sin'] = np.sin(2 * np.pi * minutes / 1440)
test['dep_cos'] = np.cos(2 * np.pi * minutes / 1440)

test= test.drop(columns=['departure_time'])

y = forest.predict(test)

pred = np.array(y)
pred = np.rint(pred).astype(int)


submission = pd.DataFrame({
    'SampleID' : sample_id,
    'delay_minutes' : pred
})
submission.to_csv('submission.csv',index=False)
#imports
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

#dataset
dataset = pd.read_csv('./train.csv')
print(f"{dataset.head()}\n")
print(f"{dataset.tail()}\n")
print(f"{dataset.info()}\n")
print(f"{dataset.describe()}\n")
print(f"{dataset.isnull().sum()}\n")

X = dataset.drop(labels=(['SampleID','temperature']),axis=1)
y = dataset['temperature']

X['day_sin'] = np.sin(2 * np.pi * X['day_of_year']/365)
X['day_cos'] = np.cos(2 * np.pi * X['day_of_year']/365)
X.drop(labels=('day_of_year'),axis=1,inplace=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)

#model
forest = RandomForestRegressor()
forest.fit(X_train, y_train)

#eval
y_pred = forest.predict(X_test)
print(f"RMSE: {root_mean_squared_error(y_test,y_pred)}\n")

#pred
test = pd.read_csv('./test.csv')

sample_id = test['SampleID']
X = test.drop(labels=('SampleID'),axis=1)
X['day_sin'] = np.sin(2 * np.pi * X['day_of_year']/365)
X['day_cos'] = np.cos(2 * np.pi * X['day_of_year']/365)
X.drop(labels=('day_of_year'),axis=1,inplace=True)

y = forest.predict(X)

submission = pd.DataFrame({
    'SampleID' : sample_id,
    'temperature'  : y
})

submission.to_csv('submission.csv',index=False)
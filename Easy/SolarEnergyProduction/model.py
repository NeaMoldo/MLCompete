#import
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

X = dataset.drop(labels=(['SampleID','energy_output']),axis=1)
y = dataset['energy_output']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)

#model
regressor = RandomForestRegressor()
regressor.fit(X_train,y_train)

#eval
y_pred = regressor.predict(X_test)
print(f"RMSE Score: {root_mean_squared_error(y_test,y_pred)}\n")

#submission
test = pd.read_csv('./test.csv')

X = test.drop(labels=('SampleID'),axis=1)
sample_id = test['SampleID']

y = regressor.predict(X)

submission = pd.DataFrame({
    'SampleID' : sample_id,
    'energy_output' : y
})

submission.to_csv('submission.csv',index=False)
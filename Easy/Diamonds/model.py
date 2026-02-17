#imports
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor

#train
train = pd.read_csv('./train.csv')

color_order = ['D','E','F','G','H','I','J']
clarity_order = ['FL', 'IF', 'VVS1', 'VVS2', 'VS1', 'VS2', 'SI1', 'SI2', 'I1', 'I2', 'I3']
cut_order = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']

encoder = OrdinalEncoder(categories=[cut_order, color_order, clarity_order])

train[['cut','color','clarity']] = encoder.fit_transform(train[['cut','color','clarity']])

train.drop(labels=('SampleID'),axis=1,inplace=True)

train['volume'] = train['x'] * train['y'] * train['z']
train['depth_table_ratio'] = train['depth'] / train['table']
train['carat_volume_ratio'] = train['carat'] / (train['volume'] + 1e-6)
X = train.drop(labels=('price'),axis=1)
y = train['price']

#test
test = pd.read_csv('./test.csv')

test[['cut','color','clarity']] = encoder.transform(test[['cut','color','clarity']])

#TASK1
task1 = test.copy()
task1['subtaskID'] = 1
task1['datapointID'] = task1['SampleID']

def byWeight(carat):
    if carat < 0.5:
        return 'Light'
    elif carat < 1.5:
        return 'Medium'
    else:
        return 'Heavy'

task1['answer'] = (
    task1['carat']
    .apply(byWeight)
)

task1 = task1[['subtaskID', 'datapointID', 'answer']]

#TASK2
task2 = test.copy()
task2['subtaskID'] = 2
task2['datapointID'] = task2['SampleID']


task2['answer'] = task2['depth']/task2['table']

task2 = task2[['subtaskID', 'datapointID', 'answer']]

#TASK3
task3 = test.copy()
task3['subtaskID'] = 3
task3['datapointID'] = task3['SampleID']

cols = ['x','y','z']
task3['answer'] = task3[cols].prod(axis=1)

task3 = task3[['subtaskID', 'datapointID', 'answer']]

#TASK4
forest = RandomForestRegressor(
    n_estimators=300,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=10
)

y_log = np.log1p(y)
forest.fit(X,y_log)

task4 = test.copy()
task4['subtaskID'] = 4
task4['datapointID'] = task4['SampleID']

task4['volume'] = task4['x'] * task4['y'] * task4['z']
task4['depth_table_ratio'] = task4['depth'] / task4['table']
task4['carat_volume_ratio'] = task4['carat'] / (task4['volume'] + 1e-6)

task4['answer'] = np.expm1(forest.predict(task4.drop(labels=(['SampleID','subtaskID','datapointID']),axis=1)))

task4 = task4[['subtaskID', 'datapointID', 'answer']]

#submission
submission = pd.concat([task1,task2,task3,task4])

submission.columns = ['subtaskID', 'datapointID', 'answer']

submission.to_csv('submission.csv',index=False)
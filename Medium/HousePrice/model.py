#imports
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import warnings

warnings.filterwarnings('ignore')

#train
train = pd.read_csv('./train.csv')

scaler = StandardScaler()

X_train = train.drop(labels=(['ID','Price']),axis=1)
y_train = train['Price']
X_train_scaled = scaler.fit_transform(X_train)

#test
test = pd.read_csv('./test.csv')

X_test =  test.drop(labels=('ID'),axis=1)
X_test_scaled = scaler.transform(X_test)

#TASK1
task1 = test.copy()
task1['subtaskID'] = 1
task1['datapointID'] = task1['ID']

task1['answer'] = task1['Square_Footage'] + task1['Garage_Size'] + task1['Lot_Size']

task1 = task1[['subtaskID', 'datapointID', 'answer']]

#TASK2
task2 = test.copy()
task2['subtaskID'] = 2
task2['datapointID'] = task2['ID']

task2['answer'] = task2['Garage_Size'] / task2['Total_Rooms']

task2 = task2[['subtaskID', 'datapointID', 'answer']]

#TASK3
task3 = test.copy()
task3['subtaskID'] = 3
task3['datapointID'] = task3['ID']

task3['answer'] = (task3['Solar_Exposure_Index'] - task3['Vibration_Level'])/task3['Magnetic_Field_Strength']

task3 = task3[['subtaskID', 'datapointID', 'answer']]

#TASK4
task4 = test.copy()
task4['subtaskID'] = 4
task4['datapointID'] = task4['ID']

task4['answer'] = abs(task4['Square_Footage'] - train['Square_Footage'].mean())

task4 = task4[['subtaskID', 'datapointID', 'answer']]

#TASK5
task5 = test.copy()
task5['subtaskID'] = 5
task5['datapointID'] = task5['ID']

forest = RandomForestRegressor(
    n_estimators=250
)
forest.fit(X_train,y_train)

task5['answer'] = forest.predict(X_test)

task5 = task5[['subtaskID', 'datapointID', 'answer']]

#submission
submission = pd.concat([task1,task2,task3,task4,task5])

submission.columns = ['subtaskID','datapointID', 'answer']

submission.to_csv('submission.csv',index=False)
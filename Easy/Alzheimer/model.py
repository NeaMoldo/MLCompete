#imports
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import warnings

warnings.filterwarnings('ignore')

#train
train = pd.read_csv('./train.csv')

X_train = train.drop(labels=(['PatientID','Diagnosis']),axis=1)
y_train = train['Diagnosis']

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

#test
test = pd.read_csv('./test.csv')

X_test = test.drop(labels=('PatientID'),axis=1)
sample_id = test['PatientID']

X_test_scaled = scaler.transform(X_test)

#TASK1
task1 = test.copy()
task1['subtaskID'] = 1
task1['datapointID'] = task1['PatientID']

age_counts = train['Age'].value_counts()
task1['answer'] = task1['Age'].map(age_counts).fillna(0).astype(int)

task1 = task1[['subtaskID', 'datapointID', 'answer']]

#TASK2
task2 = test.copy()
task2['subtaskID'] = 2
task2['datapointID'] = task2['PatientID']

age_groups = train.groupby('Age')
smoker_percent = (age_groups['Smoking'].mean() * 100)
task2['answer'] = task2['Age'].map(smoker_percent).fillna(0).round(2)

task2 = task2[['subtaskID', 'datapointID', 'answer']]

#TASK3
forest = RandomForestClassifier()
forest.fit(X_train_scaled, y_train)

task3 = test.copy()
task3['subtaskID'] = 3
task3['datapointID'] = task3['PatientID']

task3['answer'] = forest.predict_proba(X_test_scaled)[:,1]

task3 = task3[['subtaskID', 'datapointID', 'answer']]

#submission

submission = pd.concat([task1,task2,task3])

submission.columns = ['subtaskID', 'datapointID', 'answer']

submission.to_csv('submission.csv',index=False)
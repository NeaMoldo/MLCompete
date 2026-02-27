#imports
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

#train
train = pd.read_csv('./train.csv')

print(f"{train.describe()}\n")
print(f"{train.info()}\n")

train = pd.get_dummies(train, columns=["Bradykinesia","Constipation","PosturalInstability","Rigidity","SleepDisorders","Tremor"], drop_first=True)

X_train = train.drop(labels=(['PatientID','DoctorInCharge','EducationLevel','Ethnicity','EyeColor','Diagnosis']),axis=1)
y_train = train['Diagnosis']

#test
test = pd.read_csv('./test.csv')

test = pd.get_dummies(test, columns=["Bradykinesia","Constipation","PosturalInstability","Rigidity","SleepDisorders","Tremor"], drop_first=True)
test = test.drop(labels=(['DoctorInCharge','EducationLevel','Ethnicity','EyeColor']),axis=1)

#TASK1
task1 = test.copy()
task1['subtaskID'] = 'Task1'

task1['Answer'] = (task1['Hypertension']==1).astype(int) + (task1['Diabetes']==1).astype(int) + (task1['BMI']>30).astype(int)

task1 = task1[['PatientID', 'subtaskID', 'Answer']]

#TASK2
task2 = test.copy()
task2['subtaskID'] = 'Task2'

task2['Answer'] = (task2['Smoking']==1).astype(int) + (task2['AlcoholConsumption'] > 2).astype(int) + (task2['PhysicalActivity'] < 1).astype(int)

task2 = task2[['PatientID', 'subtaskID', 'Answer']]

#TASK3
forest = RandomForestClassifier(
    n_estimators=500
)
forest.fit(X_train,y_train)

task3 = test.copy()
task3['subtaskID'] = 'Task3'

task3['Answer'] = forest.predict(test.drop(labels=('PatientID'),axis=1))

task3 = task3[['PatientID', 'subtaskID', 'Answer']]

#submission
submission = pd.concat((task1,task2,task3))

submission.columns = ['PatientID', 'subtaskID', 'Answer']

submission.to_csv('submission.csv',index=False)
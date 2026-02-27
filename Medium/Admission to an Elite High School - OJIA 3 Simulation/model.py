#imports
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import warnings

warnings.filterwarnings('ignore')

#train
train  = pd.read_csv('./train.csv')

print(f"{train.info()}\n")
print(f"{train.describe()}\n")
print(f"{train.isnull().sum()}\n")

train = train.drop(labels=(['id','gen','judet']),axis=1)

X = train.drop(labels=('status_admitere'),axis=1)
y = train['status_admitere']

#test

test = pd.read_csv('./test.csv')

test = test.drop(labels=(['gen','judet']),axis=1)

#TASK1
task1 = test.copy()
task1['subtaskID'] = 1
task1['datapointID'] = task1['id']

task1['answer'] = round(task1['NT']-task1['MEV'])

task1 = task1[['subtaskID','datapointID','answer']]

#TASK2
task2 = test.copy()
task2['subtaskID'] = 2
task2['datapointID'] = task2['id']

task2['answer'] = task2['MEV'].rank(ascending=False,method='min').astype(int)

task2 = task2[['subtaskID','datapointID','answer']]

#TASK3
task3 = test.copy()
task3['subtaskID'] = 3 
task3['datapointID'] = task3['id']

#scaler = StandardScaler()
X = X.drop(labels=(['MATE','MGIM','MEV']),axis=1)
#X = scaler.fit_transform(X)

forest  = RandomForestClassifier(n_estimators=300,class_weight='balanced',random_state=10)
forest.fit(X,y)

LogReg = LogisticRegression()
LogReg.fit(X,y)

features = task3.drop(labels=(['id','subtaskID','datapointID','MATE','MGIM','MEV']),axis=1)
#features = scaler.transform(features)

task3['answer'] = LogReg.predict(features)

task3 = task3[['subtaskID','datapointID','answer']]

#submission
submission = pd.concat((task1,task2,task3))

submission.columns = ['subtaskID','datapointID','answer']

submission.to_csv('submission.csv',index=False)
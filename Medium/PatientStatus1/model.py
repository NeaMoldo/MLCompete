#imports
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import AdaBoostClassifier
import warnings

warnings.filterwarnings('ignore')

#train
train = pd.read_csv('./train.csv')

print(f'{train.describe()}\n')
print(f'{train.info()}\n')
print(f'{train.isnull().sum()}\n')

train['Tumor Size'].fillna(train['Tumor Size'].mean(),inplace=True)
train.dropna(inplace=True)

print(f'{train.isnull().sum()}\n')

train.drop(labels=(['ID','Race','Marital Status']),axis=1,inplace=True)

encoder = OrdinalEncoder()
train[['T Stage', 'N Stage', '6th Stage', 'differentiate', 'Grade', 'A Stage', 'Estrogen Status', 'Progesterone Status', 'T_N_Stage', 'Hormone_Status']] = encoder.fit_transform(train[['T Stage', 'N Stage', '6th Stage', 'differentiate', 'Grade', 'A Stage', 'Estrogen Status', 'Progesterone Status', 'T_N_Stage', 'Hormone_Status']])

train['Status'] = train['Status'].map({'Dead':0,'Alive':1})

X = train.drop(labels=('Status'),axis=1)
y = train['Status']

#test
test = pd.read_csv('./test.csv')

test.drop(labels=(['Race','Marital Status']),axis=1,inplace=True)
test[['T Stage', 'N Stage', '6th Stage', 'differentiate', 'Grade', 'A Stage', 'Estrogen Status', 'Progesterone Status', 'T_N_Stage', 'Hormone_Status']] = encoder.transform(test[['T Stage', 'N Stage', '6th Stage', 'differentiate', 'Grade', 'A Stage', 'Estrogen Status', 'Progesterone Status', 'T_N_Stage', 'Hormone_Status']])


#TASK1
task1 = test.copy()
task1['subtaskID'] = 1
task1['datapointID'] = task1['ID']

task1['answer'] = np.where(task1['GFR'] >= 90, 'Normal', 'Mildly Decreased')
task1 = task1[['subtaskID', 'datapointID', 'answer']]

#TASK2
task2 = test.copy()
task2['subtaskID'] = 2
task2['datapointID'] = task2['ID']

q1 = train['Serum Creatinine'].quantile(0.25)
q2 = train['Serum Creatinine'].quantile(0.5)
q3 = train['Serum Creatinine'].quantile(0.75)

def quantiles(x):
    if x <= q1:
        return 'Very Low'
    if x <= q2:
        return 'Low'
    if x <= q3:
        return 'High'
    else:
        return 'Very High'

task2['answer'] = task2['Serum Creatinine'].apply(quantiles)

task2= task2[['subtaskID', 'datapointID', 'answer']]

#TASK3
task3 = test.copy()
task3['subtaskID'] = 3
task3['datapointID'] = task3['ID']

bmi = train['BMI'].median()

task3['answer'] = np.where(task3['BMI'] > bmi, 1, 0)

task3 = task3[['subtaskID', 'datapointID', 'answer']]

#TASK4
task4 = test.copy()
task4['subtaskID'] = 4
task4['datapointID'] = task4['ID']

task4['answer'] = task4['T Stage'].map(train['T Stage'].value_counts()).fillna(0).astype(int)

task4 = task4[['subtaskID', 'datapointID', 'answer']]

#TASK5
task5 = test.copy()
task5['subtaskID'] = 5
task5['datapointID'] = task5['ID']

ada = AdaBoostClassifier(n_estimators=150,learning_rate=0.5,random_state=10)

ada.fit(X,y)

task5['answer'] = ada.predict(task5.drop(labels=(['ID','subtaskID','datapointID']),axis=1))

task5 = task5[['subtaskID', 'datapointID', 'answer']]

#submission
submission = pd.concat((task1,task2,task3,task4,task5))

submission.columns = ['subtaskID','datapointID','answer']

submission.to_csv('submission.csv',index=False)
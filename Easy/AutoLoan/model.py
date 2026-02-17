#imports
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

#train
train = pd.read_csv('./train.csv')

print(f"{train.head()}\n")
print(f"{train.tail()}\n")
print(f"{train.info()}\n")
print(f"{train.describe()}\n")
print(f"{train.isnull().sum()}\n")

train.drop(labels=('customer_id'),axis=1,inplace=True)

train = pd.get_dummies(train, columns=(['occupation_status','product_type','loan_intent']), drop_first=True)

X = train.drop(labels=('loan_status'),axis=1)
y = train['loan_status']

#test
test = pd.read_csv('./test.csv')

test = pd.get_dummies(test, columns=(['occupation_status','product_type','loan_intent']), drop_first=True)

#TASK1
task1 = test.copy()
task1['subtaskID'] = 1
task1['datapointID'] = task1['customer_id']

def byAge(age):
    if age < 30:
        return 'Young'
    elif age < 60:
        return 'Adult'
    else:
        return 'Senior'
 
task1['answer'] = (
    task1['age']
    .apply(byAge)
)

task1 = task1[['subtaskID', 'datapointID', 'answer']]

#TASK2
task2 = test.copy()
task2['subtaskID'] = 2
task2['datapointID'] = task2['customer_id']

def risk(dti):
    if dti < 20:
        return 'LowRisk'
    elif dti < 40:
        return 'MediumRisk'
    else:
        return 'HighRisk'
    
task2['answer'] = (
    task2['debt_to_income_ratio']
    .apply(risk)
)

task2 = task2[['subtaskID', 'datapointID', 'answer']]

#TASK3
task3 = test.copy()
task3['subtaskID'] = 3
task3['datapointID'] = task3['customer_id']

cols = ['current_debt' , 'derogatory_marks' , 'delinquencies_last_2yrs']
task3['answer'] = task3[cols].sum(axis=1)

task3 = task3[['subtaskID', 'datapointID', 'answer']]

#TASK4
forest = RandomForestClassifier()
forest.fit(X, y)

task4 = test.copy()
task4['subtaskID'] = 4
task4['datapointID'] = task4['customer_id']

test.drop(labels=('customer_id'),axis=1,inplace=True)
pred = forest.predict_proba(test)[:, 1]

task4['answer'] = pred
task4 = task4[['subtaskID', 'datapointID', 'answer']]

#submission
submission = pd.concat(objs=[task1,task2,task3,task4])

submission.columns = ['subtaskID', 'datapointID', 'answer']

submission.to_csv('submission.csv', index=False)
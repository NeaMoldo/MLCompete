#imports
import pandas as pd
import re
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
import warnings

warnings.filterwarnings('ignore')

#train
train = pd.read_csv('./train.csv')

print(f'{train.info()}\n')
print(f'{train.describe()}\n')
print(f'{train.isnull().sum()}\n')

#test
test = pd.read_csv('./test.csv')

print(f'{test.isnull().sum()}\n')

#Task 1
answer = test['NT'] - test['MEV']

task_1 = pd.DataFrame({
    'subtaskID':1,
    'datapointID':test['id'],
    'answer':answer
})

#Task 2
answer = test['MEV'].rank(ascending=False,method='min')

task_2 = pd.DataFrame({
    'subtaskID':2,
    'datapointID':test['id'],
    'answer':answer
})

#Task 3
#preprocessing
X_train = train.drop(columns=(['id','status_admitere','gen','judet']))
y_train = train['status_admitere']
X_test = test.drop(columns=(['id','gen','judet']))

scaler = StandardScaler()

cols = ['NT','MEV','MATE','MGIM']
for c in cols:
    X_train[c] = scaler.fit_transform(X_train[[c]])
    X_test[c] = scaler.transform(X_test[[c]])

X_train.columns = [re.sub(r'[":,\[\]{}]', '_', str(col)) for col in X_train.columns]
X_test.columns = [re.sub(r'[":,\[\]{}]', '_', str(col)) for col in X_test.columns]

#model
cat = CatBoostClassifier(
    iterations=2000,
    learning_rate=0.25,
    max_depth=8,
    loss_function='Logloss',
    eval_metric='F1',
    random_state=1
)
# cat.fit(X_train,y_train)

lgbmc = LGBMClassifier(
    learning_rate=0.1,
    objective='binary',
    metric='binary_logloss',
    num_iterations=1000,
    max_depth=8
)
lgbmc.fit(X_train,y_train)

answer = lgbmc.predict(X_test)

task_3 = pd.DataFrame({
    'subtaskID':3,
    'datapointID':test['id'],
    'answer':answer
})

#submission
submission = pd.concat((task_1,task_2,task_3))

submission.columns = ['subtaskID','datapointID','answer']

submission.to_csv('submission.csv',index=False)
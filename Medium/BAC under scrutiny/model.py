#imports
import pandas as pd

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
df = test[(test['liceu'] == 'COLEGIUL NATIONAL "UNIREA" FOCSANI') & (test['an'] == 2023)
          & (test['materie'] != 'LIMBA ROMANA') & (test['materie'] != 'GENERAL')]

answer = df.sort_values(by='preferinta_materie',ascending=False).iloc[0]['materie']

task_1 = pd.DataFrame({
    'subtaskID':[1],
    'datapoinID':[1],
    'answer':answer
})

#Task 2
df = test[(test['liceu'] == 'COLEGIUL NATIONAL "UNIREA" FOCSANI')
          & (test['materie'] == ' INFORMATICA MI C-C++')
          & (test['an'] >= 2014)
          & test['an'] <= 2022
          ]

answer = df.sort_values(by='preferinta_materie',ascending=False).iloc[0]['an']

task_2 = pd.DataFrame({
    'subtaskID':[2],
    'datapointID':[2],
    'answer':answer
})

#Task 3
X_train_raw = train.drop(columns=(['id','judet','medie']))
X_train_raw = X_train_raw[X_train_raw['anomalie']==0]
X_train_raw.drop(columns=('anomalie'),inplace=True)

y_train = train['medie']

X_test_raw = test.drop(columns=(['id','judet']))

X_train_raw = X_train_raw.sort_values(by=['liceu','materie','an'])


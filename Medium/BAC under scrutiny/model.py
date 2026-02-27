#imports
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
import warnings

warnings.filterwarnings('ignore')

#train
train = pd.read_csv('./train.csv')

print(f'{train.info()}\n')
print(f'{train.describe()}\n')
print(f'{train.isnull().sum()}\n')

#test
test = pd.read_csv('./test.csv')

#TASK1
unirea = test[test['liceu'] == 'COLEGIUL NATIONAL "UNIREA" FOCSANI']
unirea = unirea[(unirea['materie'] != 'LIMBA ROMANA') and (unirea['materie'] != 'GENERAL')]
answer = unirea['preferinta_materie'].max()

task_1 = pd.DataFrame({
    'id':[1],
    'subtaskID':[1],
    'answer':[answer]
})


#TASK2
bac_info = unirea[unirea['materie'] == 'INFORMATICA MI C-C++']
max_perc = unirea['preferinta_materie'].max()
sample = unirea[unirea['preferinta_materie'] == max_perc]
answer = sample['an']

task_2 = pd.DataFrame({
    'id':[2],
    'subtaskID':[2],
    'answer':[answer]
})

#TASK3
X_train = train.drop(labels=(['anomalie','medie','id']),axis=1)
y_train = train['medie']

oneHot = OneHotEncoder()
features = ['materie','liceu','judet']
X_train_encoded = oneHot.fit_transform(X_train[features])

X_test_encoded = oneHot.transform(test[features])
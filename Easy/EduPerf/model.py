#imports
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
import warnings

warnings.filterwarnings('ignore')

#train
train = pd.read_csv('./train.csv')

print(f'{train.head()}\n')
print(f'{train.info()}\n')
print(f'{train.describe()}\n')
print(f'{train.isnull().sum()}\n')

#test
test = pd.read_csv('./test.csv')

#task1
number = (test['School Type'] == 'Elementary').sum()

data = {
    'subtaskID':1,
    'datapointID':1,
    'answer':number
}

task1 = pd.DataFrame(data,index=[0])

#task2
value = test['Academic Growth Rating'].mode()[0]

data = {
    'subtaskID':2,
    'datapointID':2,
    'answer':value
}

task2 = pd.DataFrame(data,index=[0])

#task3
to_be_encoded = ['County',
                 'Alternative Education Accountability',
                 'Charter',
                 'Overall Rating',
                 'Student Achievement Rating',
                 'School Progress Rating',
                 'Academic Growth Rating',
                 'Closing the Gaps Rating'
]
ordinal_encoder = OrdinalEncoder()
train[to_be_encoded] = ordinal_encoder.fit_transform(train[to_be_encoded])

target_encoder = OrdinalEncoder()
train['Relative Performance Rating'] = target_encoder.fit_transform(train[['Relative Performance Rating']])

to_be_dropped = ['SampleID',
                 'School Type',
                 'Number of Students',
                 'County',
                 'Charter',
                 'Alternative Education Accountability',
                 'Economically Disadvantaged',
                 'EB/EL Students'
]
train.drop(labels=(to_be_dropped),axis=1,inplace=True)

X_train = train.drop(labels=(['Relative Performance Rating']),axis=1)
y_train = train['Relative Performance Rating']

test[to_be_encoded] = ordinal_encoder.transform(test[to_be_encoded])
X_test = test.drop(labels=(to_be_dropped),axis=1)

"""
mutual_info = mutual_info_classif(X_train,y_train)
mutual_info = pd.Series(mutual_info)
mutual_info.index = X.columns
mutual_info.sort_values(ascending=False)
print(mutual_info)
"""

forest = RandomForestClassifier()
forest.fit(X_train,y_train)

prediction = forest.predict(X_test)
prediction = prediction.reshape(-1,1)

task3 = pd.DataFrame(
    {
        'subtaskID':3,
        'datapointID': test['SampleID'],
        'answer':target_encoder.inverse_transform(prediction).flatten()
    }
)

#submission
submission = pd.concat((task1,task2,task3))

submission.columns = ['subtaskID','datapointID','answer']

submission.to_csv('submission.csv',index=False)
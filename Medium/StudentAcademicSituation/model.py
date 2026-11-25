#imports
import pandas as pd
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier
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

#preprocessing
X_train = train.drop(columns=(['SampleID','Target']))
y_train = train['Target']
X_test = test.drop(columns=('SampleID'))

cols_to_drop = ['Application mode',
                'Application order',
                'Gender'
                ]

X_train = X_train.drop(columns=(cols_to_drop))
X_test = X_test.drop(columns=(cols_to_drop))

scaler = StandardScaler()

cat = CatBoostClassifier(
    iterations=2000,
    learning_rate=0.05,
    max_depth=6,
    loss_function='MultiClass',
    random_state=1,
    verbose=100
)
cat.fit(X_train,y_train)

answer = cat.predict(X_test)

submission = pd.DataFrame({
    'SampleID':test['SampleID'],
    'Target':answer.flatten()
})

submission.to_csv('submission.csv',index=False)
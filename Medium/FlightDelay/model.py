#imports
import pandas as pd
from catboost import CatBoostRegressor
import warnings

warnings.filterwarnings('ignore')

#train
train = pd.read_csv('./train.csv')

print('==========TRAIN DATASET==========')
print(f'{train.info()}\n')
print(f'{train.describe()}\n')
print(f'{train.isnull().sum()}\n')

#test
test = pd.read_csv('./test.csv')

print('==========TEST DATASET==========')
print(f'{test.info()}\n')
print(f'{test.describe()}\n')
print(f'{test.isnull().sum()}\n')

#preprocessing

def preprocess(df: pd.DataFrame, is_train=True):
    if is_train:
        df = df.dropna(axis=0)
    else:
        df['airport'] = df['airport'].fillna(0)
        df = df.fillna(df.mean(numeric_only=True))

    df = df.drop(columns=['carrier_name','airport_name','year','month'])

    return df

X_train = preprocess(train, is_train=True)
y_train = X_train['delay']
X_train = X_train.drop(columns=('delay'))

X_test = preprocess(test, is_train=False)

#model
cat_f = ['carrier','airport']

cat = CatBoostRegressor(
    iterations=2500,
    depth=4,
    learning_rate=0.12,
    eval_metric='MAE',
    cat_features=cat_f,
    random_seed=1
)

cat.fit(X_train,y_train)

pred = cat.predict(X_test)

#submission
submission = pd.DataFrame({
    'sample_id':test['sample_id'],
    'delay':pred.astype(int)
})

submission.to_csv('submission.csv',index=False)
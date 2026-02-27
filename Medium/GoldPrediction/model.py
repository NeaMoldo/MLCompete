#imports
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import warnings 

warnings.filterwarnings('ignore')

#train
train = pd.read_csv('./train.csv')

print('========TRAIN DATASET==========')
print(f'{train.info()}\n')
print(f'{train.describe()}\n')
print(f'{train.isnull().sum()}\n')

#test
test = pd.read_csv('./test.csv')

print('========TEST DATASET==========')
print(f'{test.info()}\n')
print(f'{test.describe()}\n')
print(f'{test.isnull().sum()}\n')

#preprocess
def preprocess(df: pd.DataFrame):

    df = df.drop(labels=(['us_rates_%','CPI','GDP']),axis=1)
    df['usd_chf'] = df['usd_chf'].fillna(df['usd_chf'].mean())
    df['eur_usd'] = df['eur_usd'].fillna(df['eur_usd'].mean())
    
    df['sp500 open-close'] = df['sp500 open'] - df['sp500 close']
    df['nasdaq open-close'] = df['nasdaq open'] - df['nasdaq close']
    df['silver open-close'] = df['silver open'] - df['silver close']
    df['oil open-close'] = df['oil open'] - df['oil close']
    df['platinum open-close'] = df['platinum open'] - df['platinum close']
    df['palladium open-close'] = df['palladium open'] - df['palladium close']
    df['gold high-low'] = df['gold high'] - df['gold low']

    df['gold_silver_ratio']   = df['gold open'] / (df['silver close'] + 1e-9)
    df['gold_oil_ratio']      = df['gold open'] / (df['oil close'] + 1e-9)
    df['gold_platinum_ratio'] = df['gold open'] / (df['platinum close'] + 1e-9)
    df['silver_oil_ratio']    = df['silver close'] / (df['oil close'] + 1e-9)


    # if df == train:
    #     df['gold open-close'] = df['gold open'] - df['gold close']
    #     df = df.drop(labels=('gold close'),axis=1)

    cols_to_drop = ['ID','date','sp500 close','nasdaq close','silver close','oil close','platinum close','palladium close',
                    'sp500 open', 'nasdaq open','silver open','oil open','platinum open','palladium open',
                    'gold high','gold low']
    df = df.drop(labels=(cols_to_drop),axis=1)

    return df

df_train = preprocess(train)
X_train = df_train.drop(labels=('gold close'),axis=1)
y_train = df_train['gold close']
X_test = preprocess(test)

#model
forest = RandomForestRegressor(
    n_estimators=500,
    criterion='squared_error',
    max_depth=None,
    min_samples_leaf=2,
    random_state=1,
    n_jobs=-1
)

forest.fit(X_train,y_train)

submission = test.copy()
submission['gold close'] = forest.predict(X_test)

submission = submission[['ID','gold close']]

submission.to_csv('submission.csv',index=False)
#imports
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
import warnings

warnings.filterwarnings('ignore')

#train
train = pd.read_csv('./train.csv')

print(f'{train.info()}\n')
print(f'{train.describe()}\n')
print(f'{train.isnull().sum()}\n')

#test
test = pd.read_csv('./test.csv')

#preprocessing
vectorizer = TfidfVectorizer(analyzer='word',ngram_range=(2,4))

print(train['Type'].unique())

train.drop(columns=('Type'),inplace=True)
test.drop(columns=('Type'),inplace=True)

train['Year'] = 2022 - train['Year']
test['Year'] = 2022 - test['Year']

preprocessor = ColumnTransformer(
    transformers=[
        ('v1',TfidfVectorizer(max_features=10000,sublinear_tf=True,ngram_range=(1,3),analyzer='word'),'Title'),
        ('v2',TfidfVectorizer(max_features=30000,sublinear_tf=True,ngram_range=(1,3),analyzer='word'),'Description')
    ]
)

y_train = train['Score']
X_train = train.drop(columns=(['SampleID','Score']))
X_test = test.drop(columns=(['SampleID']))

X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)


cat = CatBoostRegressor(
    iterations=3000,
    learning_rate=0.02,
    depth=8,
    loss_function='MAE',
    random_state=1,
    verbose=100
)

cat.fit(X_train,y_train)

# lgbm = LGBMRegressor(
#     n_estimators=500,
#     learning_rate=0.1,
#     max_depth=5,
#     num_leaves=32   
# )

# lgbm.fit(X_train,y_train)

y_pred = cat.predict(X_test)

#submission
submission = pd.DataFrame({
    'SampleID':test['SampleID'],
    'Score':y_pred
})

submission.to_csv('submission.csv',index=False)
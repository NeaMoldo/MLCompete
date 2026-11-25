#imports
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import TargetEncoder, OneHotEncoder
from catboost import CatBoostRegressor
import warnings

warnings.filterwarnings('ignore')

#train
train = pd.read_csv('./train_data.csv')

print(f'{train.info()}\n')
print(f'{train.describe()}\n')
print(f'{train.isnull().sum()}\n')

#test
test = pd.read_csv('./test_data.csv')

print(f'{test.isnull().sum()}\n')

#Task 1
df = test[(test['City A'] == 'Barlad') & (test['Weather'] == 'Fog')]
answer = len(df)
task_1 = pd.DataFrame({
    'subtaskID':[1],
    'datapointID':[1],
    'answer':[answer]
})

#preprocessing
X_train = train.drop(columns=(['ID','deliver_time']))
y_train = train['deliver_time']
X_test = test.drop(columns=('ID'))

values_city = train['City A'].unique()
values_weather = train['Weather'].unique()
print(f'Orase: {values_city}\n')
print(f'Vreme: {values_weather}\n')

preprocessor = ColumnTransformer([
    ('CityA', TargetEncoder(),['City A']),
    ('CityB', TargetEncoder(),['City B']),
    ('vreme', OneHotEncoder(drop='first'), ['Weather'])
],remainder='passthrough')

X_train = preprocessor.fit_transform(X_train,y_train)
X_test = preprocessor.transform(X_test)

#model
cat = CatBoostRegressor(
    iterations=1600,
    learning_rate=0.025,
    max_depth=5,
    eval_metric='MAE',
    random_seed=1,
    verbose=100
)
cat.fit(X_train,y_train)

answer = (cat.predict(X_test)).flatten()

task_2 = pd.DataFrame({
    'subtaskID':2,
    'datapointID':test['ID'],
    'answer':answer
})

#submission
submission = pd.concat((task_1,task_2))
submission.columns = ['subtaskID','datapointID','answer']
submission.to_csv('submission.csv',index=False)
#imports
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor
import warnings

warnings.filterwarnings('ignore')

#train
train = pd.read_csv('./train.csv')

# print(f'{train.info()}\n')
# print(f'{train.describe()}\n')
# print(f'{train.isnull().sum()}\n')

train.dropna(axis=0,inplace=True)

#test
test = pd.read_csv('./test.csv')

# print(f'{test.isnull().sum()}\n')

#preprocessing
# print(f'Genre: {train['track_genre'].unique()}\n')
# print(f'Artists: {train['artists'].unique()}\n')
# print(f'Album Name: {train['album_name'].unique()}\n')

scaler = StandardScaler()

num_cols = [
    'duration_ms',
    'explicit',
    'danceability',
    'energy',
    'key',
    'loudness',
    'mode',
    'speechiness',
    'acousticness',
    'instrumentalness',
    'liveness',
    'valence',
    'tempo',
    'time_signature'
]

for c in num_cols:
    train[c] = scaler.fit_transform(train[[c]])
    test[c] = scaler.transform(test[[c]])

y_train = train['popularity']
X_train = train.drop(columns=(['track_name','popularity','track_id']))
X_test = test.drop(columns=(['track_name','track_id']))

#model
cat_features = ['artists','album_name','track_genre']
cat = CatBoostRegressor(
    loss_function='MAE',
    random_state=1,
    verbose=False
)

param_dist = {
    'iterations': [1500,2000,2500],
    'learning_rate':[0.1,0.25,0.5],
    'depth':[4,6,8]
}

random = RandomizedSearchCV(
    estimator=cat, 
    param_distributions=param_dist,
    n_jobs=-1
)

random.fit(X_train,y_train,cat_features=cat_features)
print(f'Best paramns: {random.best_params__}')

popularity = random.predict(X_test)

#submission
submission = pd.DataFrame({
    'track_id':test['track_id'],
    'popularity':popularity
})

submission.to_csv('submission.csv',index=False)
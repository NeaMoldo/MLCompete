#imports
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
import warnings

warnings.filterwarnings('ignore')

#train
train = pd.read_csv('./train_galactic_wars.csv')

print('=============TRAIN DATASET=============')
print(f'{train.info()}\n')
print(f'{train.describe()}\n')
print(f'{train.isnull().sum()}\n')

#test
test = pd.read_csv('./test_galactic_wars.csv')

print('=============TEST DATASET=============')
print(f'{test.info()}\n')
print(f'{test.describe()}\n')
print(f'{test.isnull().sum()}\n')

#TASK1
df = test[(test['fight_planet']=='Nabira') & (test['weather_conditions']=='Snow')]
answer = len(df)

task_1 = pd.DataFrame({
    'subtaskID':[1],
    'datapointID':[1],
    'answer':answer
})

#TASK2
answer = len(train[train['weapon_calmtrooper']=='Experimental Weapon'])

train.loc[train['weapon_calmtrooper']=='Experimental Weapon', 'weapon_calmtrooper'] = 'Double Blaster'

task_2 = pd.DataFrame({
    'subtaskID':[2],
    'datapointID':[2],
    'answer':answer
})

#PREPROCESSING

cols = ['weapon_jedai', 'weapon_calmtrooper', 'injuries', 'fight_planet', 'weather_conditions']
for c in cols:
    tr = train[c].unique()
    print(f'{len(tr)} : {tr}\n')

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
scaler = StandardScaler()

X_train_raw = train.drop(columns=['winner'])
y_train = train['winner']

def preprocess(df: pd.DataFrame, is_train: bool):
    df['armour_jedai'] = df['armour_jedai'].str.replace('%', '').astype(float)
    df['armour_calmtrooper'] = df['armour_calmtrooper'].str.replace('%','').astype(float)

    df['surprise_attack'] = df['surprise_attack'].astype(int)

    df_num = df.select_dtypes(exclude='object')
    df_cat = df.select_dtypes(include='object')

    if is_train:
        df_cat_encoded = encoder.fit_transform(df_cat)
        df_num_encoded = scaler.fit_transform(df_num)
    else:
        df_cat_encoded = encoder.transform(df_cat)
        df_num_encoded = scaler.transform(df_num)

    cat_cols = [f'{col}_{cat}' for i, col in enumerate(df_cat.columns) 
                for cat in encoder.categories_[i]]

    df_num_final = pd.DataFrame(df_num_encoded, columns=df_num.columns, index=df.index)
    df_cat_final = pd.DataFrame(df_cat_encoded, columns=cat_cols, index=df.index)

    df_final = pd.concat([df_num_final,df_cat_final],axis=1)

    return df_final

X_train = preprocess(X_train_raw, is_train=True)
X_test = preprocess(test, is_train=False)

#TASK3
task_3 = test.copy()
task_3['subtaskID'] = 3
task_3['datapointID'] = task_3['FightID']

kmeans = KMeans(n_clusters=3)

task_3['answer'] = kmeans.fit_predict(X_test)

task_3 = task_3[['subtaskID', 'datapointID', 'answer']]

#TASK4
task_4 = test.copy()
task_4['subtaskID'] = 4
task_4['datapointID'] = task_4['FightID']

param_grid = {
    'n_estimators':[300,400,500],
    'max_depth':[10,20,30],
    'min_samples_split':[2,4,5]

}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(class_weight='balanced',random_state=1),
    param_grid=param_grid,
    scoring='f1_macro',
    cv=5,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

best_pr = grid_search.best_estimator_
print(f'Best params: {best_pr}\n')

task_4['answer'] = grid_search.predict(X_test)

task_4 = task_4[['subtaskID', 'datapointID', 'answer']]

#submission
submission = pd.concat((task_1,task_2,task_3,task_4))

submission.columns = ['subtaskID', 'datapointID', 'answer']

submission.to_csv('submission.csv', index=False)
#imports
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import TargetEncoder, StandardScaler
from sklearn.cluster import KMeans
from catboost import CatBoostClassifier
import warnings

warnings.filterwarnings('ignore')

#dataset
data = pd.read_csv('./data.csv')

print(f'{data.info()}\n')
print(f'{data.describe()}\n')
print(f'{data.isnull().sum()}\n')

#task 1
df = data[  (data['location'] == 'The Old Library') & 
            (data['tech_assist'] == 1)
          ]

answer = int(len(df))

task_1 = pd.DataFrame({
    'subtaskID':[1],
    'datapointID':[1],
    'answer':[answer]
})

#Task 2
df = data[
    (data['location'] == 'Crystal Cave') &
    (data['tech_assist'] == 1) &
    (
        (data['puzzle_type'] == 'Logic') |
        (data['puzzle_type'] == 'Sequence')
    )
]

answer = df['skill_level'].mean() + df['teamwork_score'].mean()
answer = round(answer, 2)

task_2 = pd.DataFrame({
    'subtaskID':[2],
    'datapointID':[1],
    'answer':[answer]
})

#Task 3
#datasets
train = data[data['style'] != -1]
X_train = train.drop(columns=(['style','RoundID']))
y_train = train['style']

test = data[data['style'] == -1]
X_test = test.drop(columns=(['style','RoundID']))

# #preprocessing
cat = ['location','puzzle_type','secret_mechanism','story_arc']
num = X_train.select_dtypes(include=['number']).columns.to_list()

for c in cat:
    vals = data[c].values
    print(f'{c} : {vals}\n')

target = TargetEncoder()
scaler = StandardScaler()

preprocessor = ColumnTransformer([
    ('cat', target, cat),
    ('num', scaler, num)
],remainder='passthrough')

X_train = preprocessor.fit_transform(X_train,y_train)
X_test  = preprocessor.transform(X_test)

# df_train = np.vstack((X_train,X_test))

#model
# kmeans = KMeans(
#     n_clusters=3,
#     random_state=1
# )
# answer = kmeans.fit_predict(df_train)

cat = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.2,
    max_depth=6,
    random_seed=1,
    verbose=100,
)
cat.fit(X_train,y_train)

answer = cat.predict(X_test)

df_final = data.copy()
df_final.loc[df_final['style'] == -1, 'style'] = answer

task_3 = pd.DataFrame({
    'subtaskID':3,
    'datapointID':df_final['RoundID'],
    'answer':df_final['style']
})

#submission
submission = pd.concat((task_1,task_2,task_3))
submission.columns = ['subtaskID','datapointID','answer']
submission.to_csv('submission.csv',index=False)
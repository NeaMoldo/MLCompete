#imports
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, TargetEncoder, OrdinalEncoder, StandardScaler
from catboost import CatBoostClassifier
from sklearn.cluster import KMeans
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

#Task 1
ratele = train.groupby('job')['deposit'].mean()
job = ratele.idxmax()

task_1 = pd.DataFrame({
    'subtaskID':[1],
    'datapointID':[1],
    'answer':[job]
})

#Task 2
answer = train['month'].value_counts().idxmax()

task_2 = pd.DataFrame({
    'subtaskID':[2],
    'datapointID':[1],
    'answer':[answer]
})


#Task 3
X_train = train.drop(columns=(['id','deposit']))
y_train = train['deposit']
X_test = test.drop(columns=('id'))

#preprocessing
cat_cols = X_train.select_dtypes(include=['str']).columns.tolist()

for c in cat_cols:
    num = X_train[c].nunique()
    print(f'{c}: {num}')

onehot = OneHotEncoder(drop='first')
target = TargetEncoder()
ordinal = OrdinalEncoder()

preprocessor = ColumnTransformer([
    # ('binary', onehot,['default','housing','loan','marital','contact','poutcome']),
    ('jobs', target, ['job']),
    ('months', ordinal, ['month'])
],remainder='passthrough').set_output(transform='pandas')

X_train = preprocessor.fit_transform(X_train,y_train)
X_test = preprocessor.transform(X_test)

cat_features = ['remainder__marital','remainder__education','remainder__default','remainder__housing','remainder__loan',
                'remainder__contact','remainder__poutcome']

#model
cat = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.2,
    max_depth=6,
    cat_features=cat_features,
    eval_metric='F1',
    loss_function='Logloss',
    random_seed=1,
    verbose=100
)

cat.fit(X_train,y_train)

answer = cat.predict(X_test)
asnwer = answer.flatten()

task_3 = pd.DataFrame({
    'subtaskID':3,
    'datapointID':test['id'],
    'answer':asnwer
})

#Task 4
num_cols = test.select_dtypes(include=['number']).columns.tolist()
X_train = train[num_cols].drop(columns=(['id']))
X_test = test[num_cols].drop(columns=('id'))
data = pd.concat((X_train,X_test),ignore_index=True)

scaler = StandardScaler()
data = scaler.fit_transform(data)
X_test = data[len(X_train):]

kmeans = KMeans(
    n_clusters=2,
    random_state=1
)

kmeans.fit(data)
answer = kmeans.predict(X_test)
answer = 1 -answer

task_4 = pd.DataFrame({
    'subtaskID':4,
    'datapointID':test['id'],
    'answer':answer
})

#submission
submission = pd.concat((task_1,task_2,task_3,task_4))
submission.columns = ['subtaskID','datapointID','answer']
submission.to_csv('submission.csv',index=False)
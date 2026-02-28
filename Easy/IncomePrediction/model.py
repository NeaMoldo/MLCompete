#imports
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings 

warnings.filterwarnings('ignore')

#train
train = pd.read_csv('./train.csv')

print('=============TRAIN DATASET=============')
print(f'{train.info()}\n')
print(f'{train.describe()}\n')
print(f'{train.isnull().sum()}\n')

#test
test = pd.read_csv('./test.csv')

print('=============TEST DATASET=============')
print(f'{test.info()}\n')
print(f'{test.describe()}\n')
print(f'{test.isnull().sum()}\n')

#TASK 1
df = train[(train['native_country'] != 'United-States') & (train['income'] == '>50K')]

answer = df['native_country'].mode()[0]

task_1 = pd.DataFrame({
    'subtaskID':[1],
    'datapointID':[1],
    'answer':answer
})

#TASK 2
occupations = df['occupation'].unique()

dict = {}

for oc in occupations:
    nr_50k = len(train[(train['occupation']==oc) & (train['income']=='>50K')])
    nr_t = len(train[train['occupation']==oc])
    if nr_t > 0:
        rata = nr_50k/nr_t
    else:
        rata = 0
    dict.update({oc:rata})

df = pd.DataFrame(list(dict.items()),columns=['occupation','rata'])

answer = df.loc[df['rata'].idxmax(), 'occupation']

task_2 = pd.DataFrame({
    'subtaskID':[2],
    'datapointID':[2],
    'answer':[answer]
})

#Prepocessing
def preprocess(df: pd.DataFrame, is_train = True):
    df = df.drop(columns=['sampleid','education','workclass','profile_description'])
    if is_train:
        df = df.dropna(axis=0)
    else:
        df = df.fillna(0)

    return df

X_train = preprocess(train, is_train=True)
y_train = X_train['income']
X_train = X_train.drop(columns=['income'])

X_test = preprocess(test, is_train=False)

cat_f = ['marital_status','occupation','relationship','race','gender','native_country']

#TASK3
cat  = CatBoostClassifier(
    iterations=2500,
    depth=4,
    learning_rate=0.12,
    eval_metric='TotalF1:average=Macro',
    cat_features=cat_f,
    random_seed=1
)

cat.fit(X_train,y_train)

task_3 = pd.DataFrame({
    'subtaskID':3,
    'datapointID':test['sampleid'],
    'answer':cat.predict(X_test)
})

#TASK4
X_test = test['profile_description']

vectorizer = TfidfVectorizer(sublinear_tf=True)
X_test = vectorizer.fit_transform(X_test)

silh_scores = []
k_range = range(2,20)

for k in k_range:
    kmeans = KMeans(n_clusters=k,random_state=1,n_init=10)
    labels = kmeans.fit_predict(X_test)
    score = silhouette_score(X_test,labels)
    silh_scores.append(score)

optimal_k = k_range[silh_scores.index(max(silh_scores))]

kmeans = KMeans(n_clusters=optimal_k,random_state=1,n_init=10)
labels = kmeans.fit_predict(X_test)

task_4 = pd.DataFrame({
    'subtaskID':4,
    'datapointID':test['sampleid'],
    'answer':labels
})

#submission
submission = pd.concat((task_1,task_2,task_3,task_4))

submission.columns = ['subtaskID','datapointID','answer']

submission.to_csv('submission.csv',index=False)
#imports
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from catboost import CatBoostRegressor
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
def media(df):
    len1 = df['sentence1'].str.len()
    len2 = df['sentence2'].str.len()

    average = (len1 + len2)/2

    return average.tolist()

average = media(test)

def label(averages):
    rezultate = []
    for x in averages:
        if x < 50:
            rezultate.append('Short')
        elif x<100:
            rezultate.append('Medium')
        else:
            rezultate.append('Long')
    return rezultate

answer = label(average)

task_1 = pd.DataFrame({
    'subtaskID':1,
    'datapointID':test['sampleID'],
    'answer':answer
})

#Task 2
def nrTot(df):
    len1 = df['sentence1'].str.split().str.len()
    len2 = df['sentence2'].str.split().str.len()

    sum = len1 + len2

    return sum.tolist()

answer = nrTot(test)

task_2 = pd.DataFrame({
    'subtaskID':2,
    'datapointID':test['sampleID'],
    'answer':answer
})

#Task 3
def diffAbs(df):
    len1 = df['sentence1'].str.len()
    len2 = df['sentence2'].str.len()

    diff = abs(len1 - len2)

    return diff.tolist()

answer = diffAbs(test)

task_3 = pd.DataFrame({
    'subtaskID':3,
    'datapointID':test['sampleID'],
    'answer':answer
})

#Task 4
#preprocessing
X_train = train.drop(columns=(['sampleID','score']))
y_train = train['score']
X_test = test.drop(columns=('sampleID'))

vectorizer = TfidfVectorizer(
    max_features=3000,
    ngram_range=(1,3),
    analyzer='word',
    stop_words='english',
    sublinear_tf=True
)

def buildFeatures(df, tfidf, fit=False):
    s1 = df['sentence1']
    s2 = df['sentence2']

    if fit:
        tfidf.fit(pd.concat([s1,s2]))

    v1 = tfidf.transform(s1)
    v2 = tfidf.transform(s2)

    cos = np.array([
        cosine_similarity(v1[i],v2[i])[0][0]
        for i in range(v1.shape[0])
    ]).reshape(-1,1)

    diff = (v1 - v2).toarray()

    prod = v1.multiply(v2).toarray()

    return np.hstack([cos,diff,prod])

X_train = buildFeatures(X_train, vectorizer, fit=True)
X_test = buildFeatures(X_test, vectorizer, fit=False)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#model
cat = CatBoostRegressor(
    iterations=2500,
    learning_rate=0.1,
    max_depth=6,
    eval_metric='MAE',
    loss_function='MAE',
    random_seed=1,
    verbose=100
)
cat.fit(X_train,y_train)

answer = cat.predict(X_test)

task_4 = pd.DataFrame({
    'subtaskID':4,
    'datapointID':test['sampleID'],
    'answer':answer
})

#submission
submission = pd.concat((task_1,task_2,task_3,task_4))
submission.columns = ['subtaskID','datapointID','answer']
submission.to_csv('submission.csv',index=False)
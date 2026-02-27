#imports
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack
import warnings

warnings.filterwarnings('ignore')

print('Imports loaded\n')

#train
train = pd.read_csv('./train.csv')

X_train = train['chirp']
y_train = train['label']

#test
test = pd.read_csv('./test.csv')

X_test = test['chirp']

features = ['subtaskID','datapointID','answer']

print('Datasets loaded\n')

#/////TASK1\\\\\
task_1 = test.copy()

task_1['subtaskID'] = 1
task_1['datapointID'] = task_1['id']

task_1['answer'] = task_1['chirp'].str.len()

task_1 = task_1[features]

print('TASK 1 completed\n')

#/////TASK2\\\\\
task_2 = test.copy()

task_2['subtaskID'] = 2
task_2['datapointID'] = task_2['id']

task_2['answer'] = task_2['chirp'].str.count('#')

task_2 = task_2[features]

print('TASK 2 completed\n')

#/////TASK3\\\\\

#vectorization
print('Starting vectorization...\n')

vectorizer = TfidfVectorizer(
   analyzer='word',
   ngram_range=(1,3),
   max_features=5000,
   min_df=2,
   max_df=0.8,
   sublinear_tf=True
)

X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)
print('Vectorization completed\n')

# #add hashtag count
# train_hashtag = train['chirp'].str.count('#').values.reshape(-1, 1)
# test_hashtag = test['chirp'].str.count('#').values.reshape(-1, 1)

# #combine the dataframes
# train_final = hstack([X_train_vect,train_hashtag])
# test_final = hstack([X_test_vect,test_hashtag])

#model
print('Initializing, fiting the model and making predictions\n')
# mnb = MultinomialNB(alpha=0.1)
# mnb.fit(X_train_vect,y_train)
# y_pred = mnb.predict_proba(X_test_vect)

LogReg = LogisticRegression()
LogReg.fit(X_train_vect,y_train)
y_pred = LogReg.predict_proba(X_test_vect)

#build task 3 dataframe
task_3 = test.copy()

task_3['subtaskID'] = 3
task_3['datapointID'] = task_3['id']

task_3['answer'] = y_pred[:,1]

task_3 = task_3[features]

print('TASK 3 completed\n')

#Submission File
submission = pd.concat((task_1,task_2,task_3))

submission.columns = features

submission.to_csv('submission.csv',index=False)
print('Done!')
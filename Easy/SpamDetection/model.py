#import
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import warnings
warnings.filterwarnings('ignore')

#train data
train = pd.read_csv('./train.csv')

train.drop(labels=('sample_id'),axis=1,inplace=True)

X = train['text']
y = train['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)

vectorizer = TfidfVectorizer(ngram_range=(1,3))
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

#test data
test = pd.read_csv('./test.csv')

X_pred = vectorizer.transform(test['text'])

#TASK 1
task1 = test.copy()
task1['subtaskID'] = 1
task1['answer'] = task1['text'].str.len()
task1['datapointID'] = task1['sample_id']
task1 = task1[["subtaskID", "datapointID", "answer"]]

#TASK2
task2 = test.copy()
task2['subtaskID'] = 2
task2['datapointID'] = task2['sample_id']
task2['answer'] = (
    task2['text']
    .str.lower()
    .str.split()
    .apply(lambda x: x.count('free'))
)
task2 = task2[["subtaskID", "datapointID", "answer"]]

#TASK 3
LogReg = LogisticRegression()
LogReg.fit(X_train,y_train)

pred = LogReg.predict_proba(X_pred)

task3 = test.copy()
task3['subtaskID'] = 3
task3['answer'] = pred[:, 1]
task3['datapointID'] = task3['sample_id']
task3 = task3[['subtaskID', 'datapointID', 'answer']]

#eval
y_score = LogReg.predict_proba(X_test)[:, 1]

print(f"ROC AUC Score: {roc_auc_score(y_test,y_score)}\n")

#submission
submission = pd.concat([task1,task2,task3])

submission.columns = ['subtaskID', 'datapointID', 'answer']

submission.to_csv('submission.csv', index=False)
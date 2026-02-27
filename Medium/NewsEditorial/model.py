#imports
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
import warnings

warnings.filterwarnings('ignore')

#train
train = pd.read_csv('./train.csv')

X_train = train['text']
y_train = train['label']

print('Training dataset loaded\n')

#test
test = pd.read_csv('./test.csv')

X_test = test['text']

print('Testing dataset loaded\n')

#vectorize
vectorizer = TfidfVectorizer(
    analyzer='word',
    ngram_range=(1,3),
    stop_words='english',
    sublinear_tf=True
)

print('Vectorization...\n')
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)
print('Vectorization completed\n')

#submission file
submission = test.copy()

#model
print('Initializing and training the model\n')
mnb = MultinomialNB(alpha=0.121)
mnb.fit(X_train_vect,y_train)

""""
forest = RandomForestClassifier(
    n_estimators=400,
    max_depth=30,
    random_state=10,
    n_jobs=-1
)
forest.fit(X_train_vect,y_train)
"""

print('Making predictions\n')
y_pred = mnb.predict(X_test_vect)

print('Creating the submission file\n')
submission['label'] = y_pred

submission = submission[['id','label']]

submission.to_csv('submission.csv',index=False)
print('Done!')
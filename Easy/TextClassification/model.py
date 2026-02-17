#imports
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score

#dataset
train = pd.read_csv('./train.csv')

print(f"{train.head()}\n")
print(f"{train.tail()}\n")
print(f"{train.info()}\n")
print(f"{train.describe()}\n")
print(f"{train.isnull().sum()}\n")

X = train['text']
y = train['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)

vectorizer = CountVectorizer()
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

#model
mnb = MultinomialNB()
mnb.fit(X_train_vect,y_train)

#eval
y_pred = mnb.predict(X_test_vect)
print(f"F1 Score: {f1_score(y_test,y_pred,average='macro')}\n")

#submission
test = pd.read_csv('./test.csv')

X = test['text']
sample_id = test['SampleID']
X_vect = vectorizer.transform(X)

y = mnb.predict(X_vect)

submission = pd.DataFrame({
    'SampleID' : sample_id,
    'label' : y
})

submission.to_csv('submission.csv',index=False)
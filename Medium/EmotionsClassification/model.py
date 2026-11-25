#imports
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from catboost import CatBoostClassifier
import warnings

warnings.filterwarnings('ignore')

#train
train = pd.read_csv('./train.csv')

#test
test = pd.read_csv('./test.csv')

#preprocessing
X_train_raw = train['text']
y_train = train['label']
X_test_raw = test['text']

vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1,3),
    analyzer='word',
    sublinear_tf=True,
    stop_words='english'   
)

X_train = vectorizer.fit_transform(X_train_raw)
X_test = vectorizer.transform(X_test_raw)

#model
cat = CatBoostClassifier(
    iterations=1700,
    learning_rate=0.2,
    max_depth=6,
    random_seed=1
)
cat.fit(X_train,y_train)

answer = cat.predict(X_test).flatten()

#submission
submission = pd.DataFrame({
    'SampleID':test['SampleID'],
    'label':answer
})
submission.to_csv('submission.csv',index=False)
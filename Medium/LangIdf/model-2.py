#imports
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
import warnings

warnings.filterwarnings('ignore')

print('Imports Loaded\n')

#train
train = pd.read_csv('./train.csv')

X_train = train['Text']
y_train = train['language']

#test
test = pd.read_csv('./test.csv')

X_test = test['Text']

print('Datasets loaded\n')

#text processing
vectorizer = TfidfVectorizer(
    analyzer='char',
    ngram_range=(2, 5),
    max_features=20000,
    sublinear_tf=True
)

X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

print('Vectorization completed\n')

#model
forest = RandomForestClassifier(
    n_estimators=300,
    max_depth=30,
    random_state=10,
    n_jobs=-1
)

voting = VotingClassifier(
    estimators=[
        ('mnb',MultinomialNB()),
        ('lg',LogisticRegression(solver='saga'))
    ],
    voting='soft'
)

m = int(input('Choose: 1.RandomForest 2.Voting '))

if m == 1:
    model = forest
else:
    model  = voting

print('Initializing and training the models\n')

model.fit(X_train_vect,y_train_encoded)


print('Making predictions\n')
y_pred = model.predict(X_test_vect)
y_pred_encoded = label_encoder.inverse_transform(y_pred)

print('Creating the submission file\n')
submission = test.copy()
submission['language'] = y_pred_encoded
submission = submission[['SampleID','language']]
submission.to_csv('submission.csv',index=False)

print('Done!')
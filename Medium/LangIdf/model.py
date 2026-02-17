#imports
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import warnings

warnings.filterwarnings('ignore')

#train
train = pd.read_csv('./train.csv')

X_train = train['Text']
y_train = train['language']

vectorizer = CountVectorizer()
X_train_vect = vectorizer.fit_transform(X_train)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_train)

print('X_train is vectorized\n')

#test
test = pd.read_csv('./test.csv')

X_test = test['Text']
X_test_vect = vectorizer.transform(X_test)

print('X_test is vectorized\n')

#model
print('Initializing and training the model\n')
forest = RandomForestClassifier()
forest.fit(X_train_vect,y_encoded)

print('Making predictions\n')
y_pred = forest.predict(X_test_vect)
y_pred_encoded = label_encoder.inverse_transform(y_pred)

print('Creating the submission file\n')
submission = test.copy()
submission['language'] = y_pred_encoded
submission = submission[['SampleID','language']]
submission.to_csv('submission.csv',index=False)

print('Done!')
#imports
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import warnings

warnings.filterwarnings('ignore')

#train
train = pd.read_csv('./train.csv')

X_train = train['Versuri']

label_encoder = LabelEncoder()

y_train = label_encoder.fit_transform(train['Autor'])

#test
test = pd.read_csv('./test.csv')

X_test = test['Versuri']

#preprocess
def preprocess_text(text):
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

X_train = train['Versuri'].apply(preprocess_text)
X_test = test['Versuri'].apply(preprocess_text)

#vectorization
vectorizer = TfidfVectorizer(
    analyzer='word',
    ngram_range=(1,3),
    max_features=5000,
    max_df=0.8,
    min_df=2,
    sublinear_tf=True,
    strip_accents=None
)

X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect =  vectorizer.transform(X_test)

#model
mnb = MultinomialNB(alpha=0.09)

mnb.fit(X_train_vect,y_train)
y_pred_encoded = mnb.predict(X_test_vect)

# LogReg = LogisticRegression()
# LogReg.fit(X_train_vect,y_train)
# y_pred_encoded = LogReg.predict(X_test_vect)

y_pred_decoded = label_encoder.inverse_transform(y_pred_encoded)

#submission
submission = pd.DataFrame({
    'Id':test['Id'],
    'Autor':y_pred_decoded
})

submission.to_csv('submission.csv',index=False)
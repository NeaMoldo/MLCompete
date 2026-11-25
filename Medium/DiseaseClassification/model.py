#imports
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
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

#preprocess
X_train = train.drop(columns=(['Patient_ID','Disease']))
y_train = train['Disease']
X_test = test.drop(columns=('Patient_ID'))

label = LabelEncoder()
y_train = label.fit_transform(y_train)


vectorizer = TfidfVectorizer(
    ngram_range=(2,4),
    sublinear_tf=True,
    max_df=0.8,
    min_df=5,
    analyzer='word',
    stop_words='english'
)

preprocessor = ColumnTransformer([
    ('text', vectorizer, 'Symptoms'),
    ('gender', OneHotEncoder(drop='first'), ['Gender'])
], remainder='passthrough')

X_train = preprocessor.fit_transform(X_train)
X_test  = preprocessor.transform(X_test)

#model
xgbc = XGBClassifier(
    eta=0.2,
    max_depth=8,
    objective='multi:softmax',
    n_estimatos=1000,
    eval_metric='merror',
    random_state=1
)

xgbc.fit(X_train,y_train)

answer = xgbc.predict(X_test)
answer = label.inverse_transform(answer)

#submission
submission = pd.DataFrame({
    'Patient_ID':test['Patient_ID'],
    'Disease':answer
})

submission.to_csv('submission.csv',index=False)
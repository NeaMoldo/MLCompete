#imports
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import f1_score
import warnings

warnings.filterwarnings('ignore')

#dataset
dataset = pd.read_csv('./train.csv')

print(f'{dataset.describe()}\n')
print(f'{dataset.info()}\n')
print(f'{dataset.isnull().sum()}\n')

X = dataset.drop(labels=['target','SampleID'],axis=1)
y = dataset['target']

#train test split
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.8,random_state=1)

#scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#model
log_rg = LogisticRegressionCV()

log_rg.fit(X_train_scaled,y_train)

#eval
print(f'F1 Score: {f1_score(y_test,log_rg.predict(X_test_scaled),average='macro')}')

#predict
test = pd.read_csv('./test.csv')

X = test.drop(labels='SampleID',axis=1)
X_scaled = scaler.transform(X)

y = log_rg.predict(X_scaled)

#submission

submission = pd.DataFrame({
    'SampleID': test['SampleID'],
    'label': y
})

submission.to_csv('submission.csv',index=False)
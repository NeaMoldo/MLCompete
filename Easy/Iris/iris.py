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
print(f"{dataset.head()}\n")
print(f"{dataset.tail()}\n")
print(f"{dataset.describe()}\n")
print(f"{dataset.info()}\n")
print(f"{dataset.isnull().sum()}\n")

X = dataset.drop(labels=['target','SampleID'],axis=1)
y = dataset['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=10)
scaler  = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#model
log_rg = LogisticRegressionCV()
log_rg.fit(X_train_scaled,y_train)

#eval
y_pred = log_rg.predict(X_test_scaled)
print(f"Macro F1 Score: {f1_score(y_test,y_pred,average='macro')}")

#predict
test = pd.read_csv('./test.csv')

sample_ids = test['SampleID']
X = test.drop(labels='SampleID',axis=1)

X_scaled = scaler.transform(X)

y = log_rg.predict(X_scaled)

submission = pd.DataFrame({
    'SampleID' : sample_ids,
    'label' : y
})

submission.to_csv('submission.csv',index=False)
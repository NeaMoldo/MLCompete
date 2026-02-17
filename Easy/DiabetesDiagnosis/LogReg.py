#imports
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import f1_score
import warnings

warnings.filterwarnings('ignore')

#dataset
dataset = pd.read_csv('./train.csv')

print(f"{dataset.head()}\n")
print(f"{dataset.tail()}\n")
print(f"{dataset.info()}\n")
print(f"{dataset.describe()}\n")
print(f"{dataset.isnull().sum()}\n")

cols = ['pregnancies',     'glucose',  'blood_pressure',  'skin_thickness',     'insulin',         'bmi']
dataset[cols] = dataset[cols].replace(0, dataset[cols].median())

#train test
scaler = StandardScaler()

X = dataset.drop(labels=(['target','SampleID']),axis=1)
y = dataset['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#model
LogReg = LogisticRegressionCV(class_weight='balanced')
LogReg.fit(X_train_scaled, y_train)

#eval
y_pred = LogReg.predict(X_test_scaled)

print(f"F1 Score: {f1_score(y_test,y_pred,average='binary')}\n")

#submission
data = pd.read_csv('./test.csv')

X = data.drop(labels='SampleID',axis=1)
sample_id = data['SampleID']

X_scaled = scaler.transform(X)

y = LogReg.predict(X_scaled)

submission = pd.DataFrame({
    'SampleID' : sample_id,
    'label' : y
})

submission.to_csv('submission.csv',index=False)
#imports
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

#dataset
dataset  = pd.read_csv('./train.csv')

print(f"{dataset.head()}\n")
print(f"{dataset.tail()}\n")
print(f"{dataset.info()}\n")
print(f"{dataset.describe()}\n")
print(f"{dataset.isnull().sum()}\n")

#train test
X = dataset.drop(labels=(['SampleID','target']),axis=1)
y = dataset['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)

#model
forest = RandomForestClassifier()
forest.fit(X_train,y_train)

#eval
y_pred = forest.predict(X_test)
print(f"F1 Score: {f1_score(y_test,y_pred,average='macro')}\n")

#submission
data = pd.read_csv('./test.csv')

sample_id = data['SampleID']
X = data.drop(labels=('SampleID'),axis=1)
y = forest.predict(X)

submission = pd.DataFrame({
    'SampleID' : sample_id,
    'label' : y
})

submission.to_csv('submission.csv',index=False)
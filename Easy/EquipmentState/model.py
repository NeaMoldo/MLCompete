#imports
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

#dataset
dataset = pd.read_csv('./train.csv')

sample_id = dataset['SampleID']
X = dataset.drop(labels=('SampleID'),axis=1)

scaler = StandardScaler()
X_train = scaler.fit_transform(X)

#model
kmeans = KMeans(n_clusters=4,random_state=10,n_init='auto')
kmeans.fit(X_train)

#predict
test = pd.read_csv('./test.csv')

test_id = test['SampleID']
X_test = test.drop(labels=('SampleID'),axis=1)
X_test_scaled = scaler.transform(X_test)

labels = kmeans.predict(X_test_scaled)

#submission
submission = pd.DataFrame({
    'SampleID' : test_id,
    'Label' : labels
})

submission.to_csv('submission.csv',index=False)
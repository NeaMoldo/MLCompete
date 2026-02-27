#imports
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans

#dataser
data = pd.read_csv('./dataset.csv')
data['timestamp'] = pd.to_datetime(data['timestamp'])

#TASK1
task1 = pd.DataFrame([[1, data['id'].nunique(), data['vehicle_type'].nunique()]], columns=['subtaskID', 'Value1', 'Value2'])

#TASK2
vehicle_locations = data.groupby('id')[['latitude', 'longitude']].median().reset_index()
db = DBSCAN(eps=0.1)
db.fit(vehicle_locations[['latitude', 'longitude']])

task2 = pd.DataFrame({
    'subtaskID': 2,
    'Value1': vehicle_locations['id'],
    'Value2': db.labels_
})

#TASK3
type10 = data[(data['vehicle_type'] == 10) & (data['timestamp'].dt.hour.isin([0, 1, 2, 3, 4]))].copy()
if len(type10) < 10:
    type10 = data[data['vehicle_type'] == 10].copy()

kmeans = KMeans(n_clusters=3,random_state=10,n_init='auto')
kmeans.fit(type10[['latitude','longitude']])

centers = kmeans.cluster_centers_

centers_sorted = centers[centers[:, 0].argsort()]

task3 = pd.DataFrame({
    'subtaskID': 3,
    'Value1': centers_sorted[:, 0], 
    'Value2': centers_sorted[:, 1]
})

submission = pd.concat((task1,task2,task3))

submission.columns = ['subtaskID','Value1','Value2']

submission.to_csv('submission.csv',index=False)
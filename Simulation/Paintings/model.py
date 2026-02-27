#imports
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor

#train
train  = pd.read_csv('./train.csv')

train = pd.get_dummies(train, columns=['canvas_size','is_oil_painting','brush_type','has_signature','is_framed','uses_gold_leaf','is_restored','dominant_warm_colors','dominant_color', 'art_period_hint', 'auction_house', 'image_quality'], drop_first=True)

X_train = train.drop(labels=(['SampleID','target_price']),axis=1)
y_train = train['target_price']

print(f"{train.info()}\n")
print(f"{train.describe()}\n")
print(f"{train.isnull().sum()}\n")

#test
test = pd.read_csv('./test.csv')


#TASK1
task1 = test.copy()
task1['subtaskID'] = 'Task1'

task1['AAS'] = (
            ((task1['stroke_density'] > 0.7).astype(int))*2 + 
            ((task1['complexity']> 0.65).astype(int))*2 + 
            (task1['uses_gold_leaf'] == 1).astype(int) + 
            (task1['has_signature'] == 1).astype(int) + (
            ((task1['num_colors']>65) & (task1['colorfulness']>0.7)).astype(int))*2 - 
            ((task1['contrast']<0.4) | (task1['brightness']<0.45) | (task1['brightness']>0.75)).astype(int)
)

task1['Answer'] = np.where(task1['AAS'] >= 5, 'Autentic', 'Incert')

task1 = task1[['SampleID', 'subtaskID', 'Answer']]

#TASK2
task2 = test.copy()
task2['subtaskID'] = 'Task2'

task2 = pd.get_dummies(task2, columns=['canvas_size','is_oil_painting','brush_type','has_signature','is_framed','uses_gold_leaf','is_restored','dominant_warm_colors','dominant_color', 'art_period_hint', 'auction_house', 'image_quality'], drop_first=True)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(task2.drop(labels=(['SampleID','subtaskID']),axis=1))

kmeans = KMeans(n_clusters=5,random_state=10,n_init='auto')
kmeans.fit(X_train_scaled)

task2['Answer'] = kmeans.predict(X_test_scaled)

task2 = task2[['SampleID', 'subtaskID', 'Answer']]

#TASK3
task3 = test.copy()
task3['subtaskID'] = 'Task3'
task3 = pd.get_dummies(task3, columns=['canvas_size','is_oil_painting','brush_type','has_signature','is_framed','uses_gold_leaf','is_restored','dominant_warm_colors','dominant_color', 'art_period_hint', 'auction_house', 'image_quality'], drop_first=True)


forest = RandomForestRegressor(
    n_estimators=1000,
    max_depth=20,
    min_samples_split=7,
    min_samples_leaf=2,
    criterion='absolute_error'
)
forest.fit(X_train,y_train)

task3['Answer'] = forest.predict(task3.drop(labels=(['SampleID','subtaskID']),axis=1))

task3 = task3[['SampleID', 'subtaskID', 'Answer']]

#submission
submission  = pd.concat((task1,task2,task3))

submission.columns = ['SampleID', 'subtaskID', 'Answer']

submission.to_csv('submission.csv',index=False)
#imports
import pandas as pd
from catboost import CatBoostClassifier
import warnings

warnings.filterwarnings('ignore')

#train
train = pd.read_csv('./train.csv')

print(f'{train.info()}\n')
print(f'{train.describe()}\n')
print(f'{train.isnull().sum()}\n')

X = train.drop(labels=('Churn'),axis=1)
y = train['Churn']

#test
test = pd.read_csv('./test.csv')

#TASK1
task_1 = test.copy()
task_1['id'] = task_1['SampleID']
task_1['subtaskID'] = 1

task_1['answer'] = (task_1['Monthly Charge'] > 70).astype(int) + (task_1['Total Extra Data Charges'] > 10).astype(int)
task_1 = task_1[['id', 'subtaskID', 'answer']]

#TASK2
task_2 = test.copy(0)
task_2['id'] = task_2['SampleID']
task_2['subtaskID'] = 2

task_2['answer'] = (task_2['Avg Speed'] < 50).astype(int) + (task_2['Ping Score'] > 80).astype(int) + (task_2['Link Quality Index'] < 30).astype(int)

task_2 = task_2[['id', 'subtaskID', 'answer']]

#TASK3
task_3 = test.copy()
task_3['id'] = task_3['SampleID']
task_3['subtaskID'] = 3

drop_features = ['Age','SampleID','City','Country','Customer ID','Gender',
                 'Lat Long','Latitude','Longitude','Married','Partner','Population',
                 'Quarter','State','Zip Code','Payment Method','Offer','Internet Type',
                 'TV Type']

features = ['Avg Monthly Long Distance Charges','Monthly Charge',
            'Number of Referrals','Premium Tech Support','Referred a Friend',
            'Satisfaction Score','Total Extra Data Charges',
            'Total Long Distance Charges','Total Refunds','Ping Score','Link Quality Index','Total Short Distance Charges','Contract']

cat_f = ['Contract']

#X_train = train[features]
X_train = train.drop(labels=(drop_features),axis=1)
X_train = X_train.drop(labels=('Churn'),axis=1)
y_train = train['Churn']

X_test = test.drop(labels=(drop_features),axis=1)

cat = CatBoostClassifier(random_seed=1,
                         learning_rate=0.12,
                         depth=4,
                         cat_features=cat_f,
                         eval_metric='AUC')

cat.fit(X_train,y_train)

task_3['answer'] = cat.predict(X_test)

task_3 = task_3[['id', 'subtaskID', 'answer']]

#submission
submission = pd.concat((task_1,task_2,task_3))

submission.columns = ['id', 'subtaskID', 'answer']

submission.to_csv('submission.csv',index=False)


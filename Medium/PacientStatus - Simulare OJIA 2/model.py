#imports
import pandas as pd
import numpy as np
import warnings
from catboost import CatBoostClassifier
warnings.filterwarnings('ignore')

#train
train = pd.read_csv('./train.csv')

print(f'{train.info()}\n')
print(f'{train.describe()}\n')
print(f'{train.isnull().sum()}\n')

#test
test = pd.read_csv('./test.csv')

print(f'{test.isnull().sum()}\n')


#NaN values
media = train['Tumor Size'].mean()
train['Tumor Size'].fillna(media, inplace=True)

cea_mai_comuna = train['6th Stage'].mode()[0]
train['6th Stage'] = (train['6th Stage'].fillna(cea_mai_comuna)).astype(str)

#Task 1
answer = train['GFR'].apply(lambda x: 'Normal' if x >= 90 else 'Mildly Decreased')
task_1 = pd.DataFrame({
    'subtaskID':1,
    'datapointID':train['ID'],
    'answer':answer
})

#Task 2
q1 = train['Serum Creatinine'].quantile(0.25)
q2 = train['Serum Creatinine'].quantile(0.5)
q3 = train['Serum Creatinine'].quantile(0.75)

conditi = [
    (train['Serum Creatinine'] <= q1),
    (train['Serum Creatinine'] <= q2),
    (train['Serum Creatinine'] <= q3),
    (train['Serum Creatinine'] > q3)
]

labels = ['Very Low', 'Low', 'High',' Very High']

answer = np.select(conditi, labels, default='Unknown')

task_2 = pd.DataFrame({
    'subtaskID':2,
    'datapointID':train['ID'],
    'answer':answer
})

#Task 3
mediana = train['BMI'].median()
answer = train['BMI'].apply(lambda x: 1 if x > mediana else 0)

task_3 = pd.DataFrame({
    'subtaskID':3,
    'datapointID':train['ID'],
    'answer':answer
})

#Task 4
def TStage(train, test):
    rezultate = []
    train_values = train['T Stage'].values

    for p in test['T Stage']:
        nr = (train_values == p).sum()
        rezultate.append(nr)

    return rezultate

answer = TStage(train, test)

task_4 = pd.DataFrame({
    'subtaskID':4,
    'datapointID':test['ID'],
    'answer':answer
})

#Task 5
#preprocessing
X_train = train.drop(columns=(['ID','Status','Race','Marital Status']))
y_train = train['Status']
X_test = test.drop(columns=(['ID','Race','Marital Status']))

cat_features = [
    'T Stage',
    'N Stage',
    '6th Stage',
    'differentiate',
    'Grade',
    'A Stage',
    'Estrogen Status',
    'Progesterone Status',
    'T_N_Stage',
    'Hormone_Status'
]

#model
cat = CatBoostClassifier(
    iterations=2000,
    learning_rate=0.2,
    max_depth=7,
    cat_features=cat_features,
    eval_metric='Precision',
    random_seed=1
)
cat.fit(X_train,y_train)

answer = cat.predict(X_test)

task_5 = pd.DataFrame({
    'subtaskID':5,
    'datapointID':test['ID'],
    'answer':answer
})

#submission
submission = pd.concat((task_1,task_2,task_3,task_4,task_5))
submission.columns = ['subtaskID','datapointID','answer']
submission.to_csv('submssion.csv',index=False)
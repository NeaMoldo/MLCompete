#imports
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from catboost import CatBoostRegressor
import warnings

warnings.filterwarnings('ignore')

#train
train = pd.read_csv('./train.csv')

print(f'{train.info()}\n')
print(f'{train.describe()}\n')
print(f'{train.isnull().sum()}\n')

#test
test = pd.read_csv('./test.csv')

print(f'{test.info()}\n')
print(f'{test.isnull().sum()}\n')

#preprocess
train.dropna(axis=0,inplace=True)

na_cols = ['Teacher_Quality','Parental_Education_Level','Distance_from_Home']

for c in na_cols:
    col_mode = test[c].mode()[0]
    test[c] = test[c].fillna(col_mode)

#Task 1
average = train['Hours_Studied'].mean()

answer = (test['Hours_Studied'] - average).abs()

task_1 = pd.DataFrame({
    'subtaskID':1,
    'datapointID':test['ID'],
    'answer':answer
})

#Task 2
answer = (test['Sleep_Hours'] < 7)

task_2 = pd.DataFrame({
    'subtaskID':2,
    'datapointID':test['ID'],
    'answer':answer
})
#Task 3

def nr_stud_scores(train, test):
    rezultate = []
    train_scores = train['Previous_Scores'].values

    for score in test['Previous_Scores']:
        num = (train_scores >= score).sum()
        rezultate.append(num)
    
    return rezultate

task_3 = pd.DataFrame({
    'subtaskID':3,
    'datapointID':test['ID'],
    'answer':nr_stud_scores(train, test)
})

#Task 4
def nr_stud_motiv(train, test):
    rezultate = []
    train_motiv = train['Motivation_Level'].values

    for lvl in test['Motivation_Level']:
        num = (train_motiv == lvl).sum()
        rezultate.append(num)
    
    return rezultate

task_4 = pd.DataFrame({
    'subtaskID':4,
    'datapointID':test['ID'],
    'answer':nr_stud_motiv(train,test)
})

#Task 5
#preprocessing
y_train = train['Exam_Score']
X_train = train.drop(columns=(['ID','Exam_Score']))
X_test = test.drop(columns=('ID'))

ord_cols = ['Parental_Involvement','Access_to_Resources',
        'Motivation_Level','Family_Income','Teacher_Quality',
        'Peer_Influence','Parental_Education_Level','Distance_from_Home']

for c in ord_cols:
    ordinal = OrdinalEncoder()
    X_train[c] = ordinal.fit_transform(X_train[[c]])
    X_test[c] = ordinal.transform(X_test[[c]])

bool_cols = ['Extracurricular_Activities','Internet_Access','Learning_Disabilities']

for c in bool_cols:
    X_train[c] = X_train[c].map({'Yes':1,'No':0})
    X_test[c] = X_test[c].map({'Yes':1,'No':0})

cat_cols = ['School_Type','Gender']

#model
cat = CatBoostRegressor(
    iterations=2000,
    cat_features=cat_cols,
    learning_rate=0.2,
    max_depth=8,
    loss_function='MAE',
    eval_metric='MAE',
    random_state=1
)
cat.fit(X_train,y_train)

answer = (cat.predict(X_test)).flatten()

task_5 = pd.DataFrame({
    'subtaskID':5,
    'datapointID':test['ID'],
    'answer':answer
})

#submission
task_1.to_csv('submission.csv', index=False)                        
task_2.to_csv('submission.csv', index=False, mode='a', header=False)
task_3.to_csv('submission.csv', index=False, mode='a', header=False)
task_4.to_csv('submission.csv', index=False, mode='a', header=False)
task_5.to_csv('submission.csv', index=False, mode='a', header=False)
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

print(f'{test.isnull().sum()}\n')

#preprocessing
y_train = train['Exam_Score']
X_train = train.drop(columns=(['SampleID','Exam_Score']))
X_test = test.drop(columns=('SampleID'))

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
    iterations=1500,
    cat_features=cat_cols,
    learning_rate=0.1,
    max_depth=6,
    loss_function='RMSE',
    eval_metric='RMSE',
    random_state=1
)
cat.fit(X_train,y_train)

answer = (cat.predict(X_test)).flatten()

#submission
submission = pd.DataFrame({
    'SampleID':test['SampleID'],
    'Exam_Score':answer
})
submission.to_csv('submission.csv',index=False)
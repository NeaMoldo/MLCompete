#imports 
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import warnings

warnings.filterwarnings('ignore')

#train
train = pd.read_csv('./train_data.csv')

print('============TRAIN DATASET============\n')
print(f'{train.info()}\n')
print(f'{train.describe()}\n')
print(f'{train.isnull().sum()}\n')

#test
test = pd.read_csv('./test_data.csv')

print('============TEST DATASET============\n')
print(f'{test.info()}\n')
print(f'{test.describe()}\n')
print(f'{test.isnull().sum()}\n')

#TASK 1-4
answer_1 = len(train)

answer_2 = len(train[train['Gender']=='male'])

answer_3 = train['Duration'].mean()

answer_4 = len(train[train['Age'] >= 75])

task_1_4 = pd.DataFrame({
    'subtaskID':[1,2,3,4],
    'datapointID':[1,1,1,1],
    'answer':[answer_1,answer_2,answer_3,answer_4]
})

#TASK 5
X_train = train.drop(columns=('Calories'))
y_train  = train['Calories']

test['Gender'] = test['Gender'].fillna('male')

X_train = pd.get_dummies(X_train, columns=['Gender'],drop_first=True)
test = pd.get_dummies(test, columns=['Gender'],drop_first=True)

X_test_5 = test[test['Subtask'] == 5]
X_test_5 = X_test_5.drop(columns=('Subtask'))

"""
param_grid = {
    'n_estimators':[400,500,600],
    'max_depth':[10,20,30],
    'min_samples_split':[2,4,6]
}

grid_search = GridSearchCV(
    estimator=RandomForestRegressor(random_state=1),
    param_grid=param_grid,
    scoring='neg_mean_absolute_error',
    n_jobs=-1
)

grid_search.fit(X_train,y_train)

best_pr = grid_search.best_estimator_
print(f'Best params: {best_pr}\n')
"""
cols = ['Body_Temp', 'Duration', 'Heart_Rate']
poly = PolynomialFeatures()

X_train_subset = X_train[cols]
X_train_poly = poly.fit_transform(X_train_subset)

X_test_5_subset = X_test_5[cols]
X_test_5_poly = poly.transform(X_test_5_subset)

lin_reg = LinearRegression()
lin_reg.fit(X_train_poly,y_train)

task_5 = pd.DataFrame({
    'subtaskID':5,
    'datapointID':X_test_5['User_ID'],
    'answer':lin_reg.predict(X_test_5_poly)
})

#TASK 6
X_test_6 = test[test['Subtask'] == 6]
X_test_6 = X_test_6.drop(columns=('Subtask'))
X_test_6_subset = X_test_6[cols]
X_test_6_poly = poly.transform(X_test_6_subset)

task_6 = pd.DataFrame({
    'subtaskID':6,
    'datapointID':X_test_6['User_ID'],
    'answer':lin_reg.predict(X_test_6_poly)
})

#submission
submission = pd.concat((task_1_4,task_5,task_6))

submission.columns = ['subtaskID', 'datapointID', 'answer']

submission.to_csv('submission.csv', index=False)
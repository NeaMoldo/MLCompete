#imports
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV,StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import warnings

warnings.filterwarnings('ignore')

#train
train = pd.read_csv('./train.csv')

print(f'{train.head()}\n')
print(f'{train.info()}\n')
print(f'{train.describe()}\n')
print(f'{train.isnull().sum()}\n')

#test
test = pd.read_csv('./test.csv')

#TASK1
answer = ((test['HomeTeam'] == 'Chelsea') | (test['AwayTeam'] == 'Chelsea')).sum()
task_1 = pd.DataFrame({
    'subtaskID':[1],
    'datapointID':[1],
    'answer':answer
})

#TASK2
task_2 = test.copy()
task_2['subtaskID'] = 2
task_2['datapointID'] = task_2['MatchID']

aggressive_list = ['AggressiveTackler', 'RiskTaker', 'HighPressure', 'ChaosInducer']

def StyleAggressionScore(styles):
    if not isinstance(styles,list):
        return 0
    
    aggresive_count = sum(1 for style in styles if style in aggressive_list)

    return (aggresive_count/ len(styles) if len(styles) > 0 else 0).round(2)

task_2['answer'] = task_2['TeamStyles'].apply(StyleAggressionScore)

task_2 = task_2[['subtaskID','datapointID','answer']]

#TASK3
task_3 = test.copy()
task_3['subtaskID'] = 3
task_3['datapointID'] = task_3['MatchID']

label_encoder = LabelEncoder()

all_teams = pd.concat([
    train['HomeTeam'], train['AwayTeam'],
    test['HomeTeam'],  test['AwayTeam']
]).unique()

label_encoder.fit(all_teams)

task_3['score'] = task_3['TeamStyles'].apply(StyleAggressionScore)
task_3['HomeTeam'] = label_encoder.transform(task_3['HomeTeam'])
task_3['AwayTeam'] = label_encoder.transform(task_3['AwayTeam'])

train['score'] = train['TeamStyles'].apply(StyleAggressionScore)
X_train = train.drop(labels=(['MatchID','Season','FullTimeResult','TeamStyles','chaos_label']),axis=1)
X_train['HomeTeam'] = label_encoder.transform(X_train['HomeTeam'])
X_train['AwayTeam'] = label_encoder.transform(X_train['AwayTeam'])
y_train = train['chaos_label']

X_test = task_3.drop(labels=(['MatchID','Season','FullTimeResult','TeamStyles','subtaskID','datapointID']),axis=1)

forest = RandomForestClassifier()

param_grid = {
    'n_estimators':[200,250,300,350,400,450,500],
    'max_features':['sqrt', 'log2'],
    'max_depth':[4, 5, 6, 7, 8],
    'criterion':['gini', 'entropy']
}

grid_search = GridSearchCV(estimator=forest,param_grid=param_grid,cv=StratifiedKFold(10),n_jobs=-1,scoring='f1_macro')
grid_search.fit(X_train,y_train)

print('Best params: ', grid_search.best_params_,'\n')

task_3['answer'] = grid_search.predict(X_test)

task_3 = task_3[['subtaskID','datapointID','answer']]

#submission
submission = pd.concat((task_1,task_2,task_3))

submission.columns = ['subtaskID','datapointID','answer']

submission.to_csv('submission.csv',index=False)
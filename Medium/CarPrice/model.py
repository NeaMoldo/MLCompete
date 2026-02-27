#imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
import warnings

warnings.filterwarnings('ignore')

#train
train = pd.read_csv('./train.csv')

print(f'{train.info()}\n')
print(f'{train.describe()}\n')
print(f'{train.isnull().sum()}\n')


#test
test = pd.read_csv('./test.csv')


#TASK1
answer = train['enginetype'].mode()[0]

task_1 = pd.DataFrame({
    'subtaskID':[1],
    'datapointID':[1],
    'answer':[answer]
})

#TASK2
gas = train[train['fueltype']=='gas']

answer = gas['price'].mean().round(2)

task_2 = pd.DataFrame({
    'subtaskID':[2],
    'datapointID':[1],
    'answer':[answer]
})

#TASK3
# df_plot = train.drop(labels=(['CarID','CarName','fueltype','aspiration','doornumber','carbody','drivewheel','enginelocation','enginetype','fuelsystem']),axis=1)

# sns.pairplot(df_plot)
# plt.savefig('pairplot.png')

train.dropna(axis=0,inplace=True)

X_train = train.drop(labels=('price'),axis=1)

def preprocess(df):
    df = df.drop(labels=(['CarID','CarName','fueltype','aspiration','doornumber','carbody','drivewheel','enginelocation','enginetype','fuelsystem','cylindernumber']),axis=1)
    df['carlength/wheelbase'] = df['carlength']/df['wheelbase']
    df['carlength/curbweight'] = df['carlength']/df['curbweight']
    df['carvolume'] = df['carlength']*df['carwidth']*df['carheight']
    df['HP/highwaympg'] = df['horsepower']/df['highwaympg']
    df['HP/citympg'] = df['horsepower']/df['citympg']
    df['highwaympg/citympg'] = df['highwaympg'] / df['citympg']

    df = df.fillna(df.mean())

    df = df.drop(labels=(['carlength','carwidth','carheight','wheelbase','curbweight']),axis=1)

    return df

X_train = preprocess(X_train)
y_train = train['price']

X_test = preprocess(test)

forest = RandomForestRegressor(n_estimators=250,
                               criterion='squared_error',
                               max_depth=4,
                               n_jobs=-1)

forest.fit(X_train,y_train)

task_3 = test.copy()
task_3['subtaskID'] = 3
task_3['datapointID'] = task_3['CarID']

task_3['answer'] = forest.predict(X_test)

task_3 = task_3[['subtaskID','datapointID','answer']]

#submission
submission = pd.concat((task_1,task_2,task_3))

submission.columns = ['subtaskID','datapointID','answer']

submission.to_csv('submission.csv',index=False)
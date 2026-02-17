#imports
import pandas as pd

#train
muzee = pd.read_csv('./muzee.csv')

print(f'{muzee.head()}\n')
print(f'{muzee.info()}\n')
print(f'{muzee.describe()}\n')
print(f'{muzee.isnull().sum()}\n')

#TASK1
answer = len(muzee)

task_1 = pd.DataFrame({
    'id':[1],
    'subtaskID':[1],
    'answer':answer
})

#TASK2
muzee_buc = muzee[muzee['județul']=='București']
answer = len(muzee_buc)

task_2 = pd.DataFrame({
    'id':[2],
    'subtaskID':[2],
    'answer':answer
})

#TASK3
answer = muzee.isna().any().sum()

task_3 = pd.DataFrame({
    'id':[3],
    'subtaskID':[3],
    'answer':answer
})

#TASK4
answer = muzee['anul înființării'].value_counts().idxmax()

task_4 = pd.DataFrame({
    'id':[4],
    'subtaskID':[4],
    'answer':answer
})

#TASK5
judete = muzee['județul'].unique().tolist()

answer_list = []

for judet in judete:
    df = muzee[muzee['județul']==judet]
    answer = len(df)
    answer_list.append(answer)

task_5 = pd.DataFrame({
    'id':judete,
    'subtaskID':5,
    'answer':answer_list
})

#TASK6
answer = muzee.count(axis=1) / 44 * 100

task_6 = pd.DataFrame({
    'id':muzee['_id'],
    'subtaskID':6,
    'answer':answer.round(2)
})

#TASK7
answer = task_6['answer'].mean().round(2)

task_7 = pd.DataFrame({
    'id':[7],
    'subtaskID':[7],
    'answer':answer
})

#TASK8
compl_mx = task_6[task_6['answer'] == task_6['answer'].max()]
nr = len(compl_mx)
answer = nr / len(muzee) * 100

task_8 = pd.DataFrame({
    'id':[8],
    'subtaskID':[8],
    'answer':round(answer,2)
})


#submission
submission = pd.concat((task_1,task_2,task_3,task_4,task_5,task_6,task_7,task_8))

submission.columns = ['id','subtaskID','answer']

submission.to_csv('submission.csv',index=False)

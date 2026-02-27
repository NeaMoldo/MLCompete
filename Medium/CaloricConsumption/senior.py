import pandas as pd

train = pd.read_csv('./train_data.csv')

answer = len(train[train['Age'] >= 75])

print(answer)
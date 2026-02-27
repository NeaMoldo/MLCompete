import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv('./train_data.csv')

train['BMI'] = train['Weight'] / (train['Height'] * train['Height'])

features = ['Age' , 'Height', 'Weight', 'Duration', 'Heart_Rate',
            'Body_Temp', 'BMI']

y = 'Calories'

for f in features:
    plt.scatter(train[y],train[f])
    plt.xlabel(y)
    plt.ylabel(f)
    plt.savefig(f'./Plots/{f}.png')
    plt.close()
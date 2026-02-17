#imports
import pandas as pd
import numpy as np

test = pd.read_csv('./test.csv')

sample_id = test['SampleID']
v_values = np.array(test['temperature_c'])

f_values = np.round(np.array(v_values * 9 / 5 + 32), decimals=2)

df = pd.Series(f_values)

submission = pd.DataFrame({
    'SampleID' : sample_id,
    'temperature_f' : df
})

submission.to_csv('submission.csv',index=False)
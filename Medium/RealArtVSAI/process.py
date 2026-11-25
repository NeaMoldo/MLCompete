import pandas as pd

sub = pd.read_csv('submission.csv')
test = pd.read_csv('test.csv')

sub['SampleID'] = test['SampleID']

sub.to_csv('submission.csv', index=False)
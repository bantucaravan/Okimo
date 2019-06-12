import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
import json
if pd.__version__ <  '0.24.0':
    raise ImportError("This script requires pandas 0.24.0 or greater")

# The script uses .shift() with fill_value= arg which as changed in version
# 0.24.0. the fill_value= arg is necessary to sets first fixation to 
# as progressive by default. Consider exploring alternate appraches with
#  the same effect to avoid dependency issues.
# Consider: leaving the NaN from shift and using .fillna() on the resulting df
# docs: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.shift.html



def main(data):

    # Create df of fixation per row
    fix_data = []
    fix_data_index = []
    #json_str = data.fixations[0]
    for test_id, json_str in zip(data.test, data.fixations):
        json_dict = json.loads(json_str)
        # there are 3 tests with no right fixations
        # and 4 tests with only one fixation in both eyes
        # Our feature framework requires one Regr and one prog fixation
        # so less than 2 fixations is default inadmissible
        if (len(json_dict['R']) > 1) and (len(json_dict['L']) > 1):
            fix_data.append(json_normalize(json_dict['L']))
            fix_data.append(json_normalize(json_dict['R']))
            fix_data_index.extend([(test_id, 'L'), (test_id, 'R')])
    fix_data = pd.concat(fix_data, axis=0, keys=fix_data_index) 
    fix_data.index.names = ['test_id', 'eye', None]

    # Identify regressive fixations ('Regr'=regressive, 'Prog'=progressive)
    # first fixation in each eye is default progressive
    def diffs(df):
        df['X.diff'] = df.X - df.X.shift(1, fill_value=df.X[0])
        df['Y.diff'] = df.Y - df.Y.shift(1, fill_value=df.Y[0])
        df['T.diff'] = df['T'] - df['T'].shift(1, fill_value=df['T'][0])
        #mask = (df.X - df.X.shift(1, fill_value=0)) < 0 
        #df['Regressive'] = 0
        #df.loc[mask, 'Regressive'] = 1
        return df #['Regressive']

    # Note: this feature framework requires at least one regressive and one
    # progressive fixation for each eye. Currently there is no check for that.
    fix_data  = fix_data.groupby(['test_id', 'eye']).apply(diffs)
    mask = fix_data['X.diff'] < 0
    fix_data['Regressive'] = 'Prog'
    fix_data.loc[mask, 'Regressive'] = 'Regr'


    ## Create df of generated features, one test per row
    def max_diff(series):
        return np.abs(series).max()
    def min_diff(series):
        series = series[series != 0]
        return np.abs(series).min()

    funcs = {'X' :['mean', 'std', len], 'Y': ['mean', 'std'],
    'X.diff': ['mean', 'std', max_diff, min_diff], 
    'Y.diff': ['mean', 'std', max_diff, min_diff]}

    feat_df = fix_data.groupby(['test_id', 'eye', 'Regressive']).agg(funcs)
    feat_df.columns = ['_'.join(tup) for tup in feat_df.columns.values]

    # Combine col headers and rearrange df shape
    feat_df = feat_df.stack().reset_index()
    lists = feat_df[['eye', 'Regressive', 'level_3']].values.tolist()
    lists = ['_'.join(i) for i in lists]
    feat_df = feat_df.drop(['eye',	'Regressive',	'level_3'], axis=1)
    feat_df['feats'] = lists
    feat_df = feat_df.pivot(index='test_id', columns='feats')
    feat_df.columns = [tup[1] for tup in feat_df.columns.values]

    return feat_df



if __name__ == '__main__':

    ## Load data
    file = './data/Reading Tests v1 + v3.csv'
    data = pd.read_csv(file)

    # Run feature generation and save to csv
    data = main(data)
    data.to_csv('./Data/FixationFeatures.csv')



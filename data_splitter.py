import pandas as pd


def split(data):
    train_start = pd.to_datetime('2013-01-22')
    test_start = pd.to_datetime('2017-01-01')
    train_data_mask = (train_start < data['Date']) & (data['Date'] < test_start)
    test_data_mask = (test_start < data['Date']) & (data['Date'] < pd.to_datetime('today'))
    train_data = data.loc[train_data_mask]
    test_data = data.loc[test_data_mask]
    return train_data, test_data

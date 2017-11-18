import pandas as pd


class DataSplitter:
    @classmethod
    def split(self, data):
        train_start = pd.to_datetime('2013-01-22')
        test_start = pd.to_datetime('2017-01-01')
        train_data_mask = (train_start <= data['date']) & (data['date'] < test_start)
        test_data_mask = (test_start <= data['date']) & (data['date'] < pd.to_datetime('today'))
        train_data = data.loc[train_data_mask]
        test_data = data.loc[test_data_mask]
        return train_data, test_data

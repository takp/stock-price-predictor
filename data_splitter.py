import pandas as pd
import numpy


class DataSplitter:
    @classmethod
    def split_to_x_and_y(self, data, timesteps):
        x, y = [], []
        for i in range(len(data) - timesteps):
            x.append(data.iloc[i:(i + timesteps)].drop('date', axis=1).as_matrix())
            y.append([data.iloc[i + timesteps]['nikkei']])
        return numpy.array(x), numpy.array(y)

    @classmethod
    def split_to_train_and_test(self, data):
        train_start = pd.to_datetime('2013-01-22')
        test_start = pd.to_datetime('2017-01-01')
        train_data_mask = (train_start <= data['date']) & (data['date'] < test_start)
        test_data_mask = (test_start <= data['date']) & (data['date'] < pd.to_datetime('today'))
        train_data = data.loc[train_data_mask]
        test_data = data.loc[test_data_mask]
        return train_data, test_data

import pandas as pd
from datetime import timedelta


class DataPreprocessor:
    def __init__(self, data):
        self.data = data

    def preprocess_data(self, dropping_features):
        data = self.data.drop(dropping_features, axis=1)
        data.columns = ['date', 'ClosePrice']
        data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')
        data = self.__calculate_change_rate(data)
        return data

    def __calculate_change_rate(self, data):
        data['change_rate'] = pd.Series()
        for i in range(len(self.data) - 1):
            close_prices = data['ClosePrice']
            change_rate = (float(close_prices[i]) / close_prices[i + 1])
            rounded_change_rate = round(change_rate, 50)
            data.set_value(i, 'change_rate', rounded_change_rate)
        return data

    @classmethod
    def merge(cls, nikkei_data, nasdaq_data, currency_data):
        df = cls.__create_blank_dataframe()
        df['nikkei'] = pd.Series()
        df['nasdaq'] = pd.Series()
        df['currency'] = pd.Series()
        # Input each data
        for i in range(len(df)):
            date = df.loc[i, 'date']
            nikkei = nikkei_data.loc[(nikkei_data['date'] == date)]
            if not nikkei.empty:
                df.loc[i, 'nikkei'] = nikkei['change_rate'].item()
            nasdaq = nasdaq_data.loc[(nasdaq_data['date'] == date)]
            if not nasdaq.empty:
                df.loc[i, 'nasdaq'] = nasdaq['change_rate'].item()
            currency = currency_data.loc[(currency_data['date'] == date)]
            if not currency.empty:
                df.loc[i, 'currency'] = currency['change_rate'].item()
        return df

    @classmethod
    def __create_blank_dataframe(cls):
        today = pd.to_datetime('today')
        start = pd.to_datetime('2003-01-22')
        dates = []
        for i in cls.__daterange(start, today):
            # print(i.strftime("%Y-%m-%d"))
            dates.append(pd.to_datetime(i))
        df = pd.DataFrame(dates, columns=['date'])
        return df

    @classmethod
    def __daterange(cls, start, end):
        for i in range(int((end - start).days)):
            yield start + timedelta(i)

import pandas as pd
from datetime import timedelta

def preprocess_nikkei(data):
    dropping_features = ['Open Price', 'High Price', 'Low Price']
    data = preprocess_data(data, dropping_features)
    return data[:len(data) - 1]


def preprocess_nasdaq(data):
    dropping_features = ['High', 'Low', 'Total Market Value', 'Dividend Market Value']
    data = preprocess_data(data, dropping_features)
    return data[:len(data) - 1]


def preprocess_currency(data):
    dropping_features = ['High (est)', 'Low (est)']
    data = preprocess_data(data, dropping_features)
    return data[:len(data) - 1]


def preprocess_data(data, dropping_features):
    data = data.drop(dropping_features, axis=1)
    data.columns = ['date', 'ClosePrice']
    data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')
    data = calculate_change_rate(data)
    return data


def calculate_change_rate(data):
    data['change_rate'] = pd.Series()
    for i in range(len(data) - 1):
        close_prices = data['ClosePrice']
        change_rate = (float(close_prices[i]) / close_prices[i + 1]) * 100
        rounded_change_rate = round(change_rate, 50)
        data.set_value(i, 'change_rate', rounded_change_rate)
    return data


def merge(nikkei_data, nasdaq_data, currency_data):
    df = create_blank_dataframe()
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


def create_blank_dataframe():
    today = pd.to_datetime('today')
    start = pd.to_datetime('2013-01-22')
    dates = []
    for i in daterange(start, today):
        # print(i.strftime("%Y-%m-%d"))
        dates.append(pd.to_datetime(i))
    df = pd.DataFrame(dates, columns=['date'])
    return df


def daterange(start, end):
    for i in range(int ((end - start).days)):
        yield start + timedelta(i)


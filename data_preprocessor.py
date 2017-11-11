import pandas as pd


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
    data.columns = ['Date', 'ClosePrice']
    data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d')
    data = calculate_change_rate(data)
    return data


def calculate_change_rate(data):
    data['ChangeRate'] = pd.Series()
    for i in range(len(data) - 1):
        close_prices = data['ClosePrice']
        change_rate = (float(close_prices[i]) / close_prices[i + 1]) * 100
        rounded_change_rate = round(change_rate, 50)
        data.set_value(i, 'ChangeRate', rounded_change_rate)
    return data

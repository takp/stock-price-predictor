import pandas as pd


def preprocess_nikkei(data):
    dropping_feature = ['Open Price', 'High Price', 'Low Price']
    new_data = data.drop(dropping_feature, axis=1)
    new_data['ChangeRate'] = pd.Series()
    for i in range(len(new_data) - 1):
        close_prices = new_data['Close Price']
        change_rate = (float(close_prices[i]) / close_prices[i + 1]) * 100
        rounded_change_rate = round(change_rate, 50)
        new_data.set_value(i, 'ChangeRate', rounded_change_rate)
    new_data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d')
    return new_data


def preprocess_nasdaq(data):
    return data


def preprocess_currency(data):
    return data

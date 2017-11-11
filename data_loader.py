import pandas as pd

nikkei_file_path = 'data/NIKKEI-INDEX.csv'
nasdaq_file_path = 'data/NASDAQOMX-COMP.csv'
jpyusd_file_path = 'data/CURRFX-USDJPY.csv'


def load_dataset():
    try:
        nikkei_data = pd.read_csv(nikkei_file_path)
        print("Nikkei dataset has {} samples with {} features each.".format(*nikkei_data.shape))
        nasdaq_data = pd.read_csv(nasdaq_file_path)
        print("Nasdaq dataset has {} samples with {} features each.".format(*nasdaq_data.shape))
        currency_data = pd.read_csv(jpyusd_file_path)
        print("Currency dataset has {} samples with {} features each.".format(*currency_data.shape))
        return nikkei_data, nasdaq_data, currency_data
    except:
        print("Dataset could not be loaded. Is the dataset missing?")

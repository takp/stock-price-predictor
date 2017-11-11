import data_loader
import data_preprocessor

# Load data
nikkei_data_org, nasdaq_data_org, currency_data_org = data_loader.load_dataset()

# Data Preprocessing
nikkei_data = data_preprocessor.preprocess_nikkei(nikkei_data_org)
nasdaq_data = data_preprocessor.preprocess_nasdaq(nasdaq_data_org)
currency_data = data_preprocessor.preprocess_currency(currency_data_org)

print(nikkei_data)
print(nasdaq_data)
print(currency_data)

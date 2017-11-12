import data_loader
import data_preprocessor
import data_splitter

# Load data
nikkei_data_org, nasdaq_data_org, currency_data_org = data_loader.load_dataset()

# Data Preprocessing
nikkei_data = data_preprocessor.preprocess_nikkei(nikkei_data_org)
nasdaq_data = data_preprocessor.preprocess_nasdaq(nasdaq_data_org)
currency_data = data_preprocessor.preprocess_currency(currency_data_org)

print(nikkei_data)
print(nasdaq_data)
print(currency_data)

# Split the data
nikkei_train_data, nikkei_test_data = data_splitter.split(nikkei_data)
print("Nikkei train dataset has {} samples.".format(*nikkei_train_data.shape))
print("Nikkei test dataset has {} samples.".format(*nikkei_test_data.shape))
# print(nikkei_train_data.head())
# print(nikkei_test_data.head())

nasdaq_train_data, nasdaq_test_data = data_splitter.split(nasdaq_data)
print("Nasdaq train dataset has {} samples.".format(*nasdaq_train_data.shape))
print("Nasdaq test dataset has {} samples.".format(*nasdaq_test_data.shape))
# print(nasdaq_train_data.head())
# print(nasdaq_test_data.head())

currency_train_data, currency_test_data = data_splitter.split(currency_data)
print("Currency train dataset has {} samples.".format(*currency_train_data.shape))
print("Currency test dataset has {} samples.".format(*currency_test_data.shape))
# print(currency_train_data.head())
# print(currency_test_data.head())

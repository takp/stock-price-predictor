import data_loader
from data_preprocessor import DataPreprocessor
import data_splitter
import lstm_model

# Load data
nikkei_data_org, nasdaq_data_org, currency_data_org = data_loader.load_dataset()

# Data Preprocessing
dropping_features_for_nikkei = ['Open Price', 'High Price', 'Low Price']
dropping_features_for_nasdaq = ['High', 'Low', 'Total Market Value', 'Dividend Market Value']
dropping_features_for_currency = ['High (est)', 'Low (est)']

nikkei_data = DataPreprocessor(nikkei_data_org).preprocess_data(dropping_features_for_nikkei)
nasdaq_data = DataPreprocessor(nasdaq_data_org).preprocess_data(dropping_features_for_nasdaq)
currency_data = DataPreprocessor(currency_data_org).preprocess_data(dropping_features_for_currency)

# Merge data
merged_data = DataPreprocessor.merge(nikkei_data, nasdaq_data, currency_data)
print(merged_data.head(20))

#
#
# # Split the data
# nikkei_train_data, nikkei_test_data = data_splitter.split(nikkei_data)
# print("Nikkei train dataset has {} samples.".format(*nikkei_train_data.shape))
# print("Nikkei test dataset has {} samples.".format(*nikkei_test_data.shape))
# # print(nikkei_train_data.head())
# # print(nikkei_test_data.head())
#
# nasdaq_train_data, nasdaq_test_data = data_splitter.split(nasdaq_data)
# print("Nasdaq train dataset has {} samples.".format(*nasdaq_train_data.shape))
# print("Nasdaq test dataset has {} samples.".format(*nasdaq_test_data.shape))
# # print(nasdaq_train_data.head())
# # print(nasdaq_test_data.head())
#
# currency_train_data, currency_test_data = data_splitter.split(currency_data)
# print("Currency train dataset has {} samples.".format(*currency_train_data.shape))
# print("Currency test dataset has {} samples.".format(*currency_test_data.shape))
# # print(currency_train_data.head())
# # print(currency_test_data.head())

# Build & train model
model = lstm_model.LSTMModel().build()
# model.fit(X_train, y_train, batch_size=10, nb_epoch=0.05)

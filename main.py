import data_loader
from data_preprocessor import DataPreprocessor
from data_splitter import DataSplitter
from lstm_model import LSTMModel

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
# Remove non-working days
data = merged_data.dropna()
# Split data to train and test by the date
data_train, data_test = DataSplitter.split_to_train_and_test(data)

x_train, y_train = DataSplitter.split_to_x_and_y(data_train, length_of_sequence=10)
x_test, y_test = DataSplitter.split_to_x_and_y(data_test, length_of_sequence=10)

print("Train dataset has {} samples.".format(*x_train.shape))
# print(x_train[0:5])
# print(y_train[0:5])
print("Test dataset has {} samples.".format(*x_test.shape))
# print(x_test[0:5])
# print(y_test[0:5])

# Build & train model
model = LSTMModel().build()
# model.fit(X_train, y_train, batch_size=10, nb_epoch=0.05)

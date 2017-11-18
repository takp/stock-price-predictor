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

# Split the data
x_train, x_test = DataSplitter.split(data)

print("Train dataset has {} samples.".format(*x_train.shape))
print(x_train.head(10))

print("Test dataset has {} samples.".format(*x_test.shape))
print(x_test.head(10))

# Build & train model
model = LSTMModel().build()
# model.fit(X_train, y_train, batch_size=10, nb_epoch=0.05)

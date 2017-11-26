import data_loader
from data_preprocessor import DataPreprocessor
from data_splitter import DataSplitter
from lstm_model import LSTMModel
from dummy_predictor import DummyPredictor
import pandas as pd
import numpy as np

# Configurations
timesteps = 10
hidden_neurons = 50
epochs = 100
batchsize = 10

# Load data
nikkei_data_org, nasdaq_data_org, currency_data_org = data_loader.load_dataset()

# Data Preprocessing
dropping_features_for_nikkei = ['Open Price', 'High Price', 'Low Price']
dropping_features_for_nasdaq = ['High', 'Low', 'Total Market Value', 'Dividend Market Value']
dropping_features_for_currency = ['High (est)', 'Low (est)']

nikkei_data = DataPreprocessor(nikkei_data_org).preprocess_data(dropping_features_for_nikkei)
nasdaq_data = DataPreprocessor(nasdaq_data_org).preprocess_data(dropping_features_for_nasdaq)
currency_data = DataPreprocessor(currency_data_org).preprocess_data(dropping_features_for_currency)

merged_data = DataPreprocessor.merge(nikkei_data, nasdaq_data, currency_data)
data = merged_data.dropna()

# Split the data
data_train, data_val, data_test = DataSplitter.split_to_train_val_test(data)
x_train, y_train = DataSplitter.split_to_x_and_y(data_train, timesteps=timesteps)
x_val, y_val = DataSplitter.split_to_x_and_y(data_val, timesteps=timesteps)
x_test, y_test = DataSplitter.split_to_x_and_y(data_test, timesteps=timesteps)

print("Train dataset has {} samples.".format(*x_train.shape))
# print(x_train[:3])
# print(y_train[:3])
print("Validation dataset has {} samples.".format(*x_val.shape))
# print(x_val[:3])
# print(y_val[:3])
print("Test dataset has {} samples.".format(*x_test.shape))
# print(x_test[:3])
# print(y_test[:3])

# Build & train model
model = LSTMModel(timesteps, hidden_neurons).build()
print("Fitting the model...")
model.fit(x_train, y_train,
          batch_size=batchsize, epochs=epochs, validation_data=(x_val, y_val))

print("Predicting...")
result = model.predict(x_test)
predicted = pd.DataFrame(result)
predicted.columns = ['predicted_nikkei']
predicted['actual_nikkei'] = y_test

print("Completed Prediction.")
print(predicted.shape)
print(predicted[:10])
# Output to csv
predicted.to_csv("predicted.csv")

# Evaluate
evaluation_score = model.evaluate(x_test, y_test, batch_size=batchsize, verbose=1)
print("Evaluation score is {}".format(evaluation_score))

# Evaluate benchmark
dummy_predicted = DummyPredictor().get_evaluation_score(y_train)
dummy_evaluation_score = np.mean((y_test - dummy_predicted) ** 2)
print("Dummy evaluation score is {}".format(dummy_evaluation_score))

# Compare
comparison = (evaluation_score / dummy_evaluation_score) * 100
print("This prediction model's MSE is {} percent compared to benchmark. (smaller is better)".format(comparison))

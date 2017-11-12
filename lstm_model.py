from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM


class LSTMModel:
    def __init__(self):
        self.length_of_sequences = 10
        self.in_out_neurons = 1
        self.hidden_neurons = 300

    def build(self):
        model = Sequential()

        model.add(LSTM(input_shape=(50, 1), return_sequences=True, units=50))
        model.add(Dropout(0.2))

        # model.add(LSTM(100, return_sequences=False))
        # model.add(Dropout(0.2))

        model.add(Dense(1))
        model.add(Activation("linear"))

        model.compile(loss="mean_absolute_percentage_error", optimizer="rmsprop")

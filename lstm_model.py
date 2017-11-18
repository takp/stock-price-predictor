from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.recurrent import LSTM


class LSTMModel:
    def __init__(self, timesteps=10, hidden_neurons=50):
        self.timesteps = timesteps
        self.hidden_neurons = hidden_neurons
        self.input_dim = 3

    def build(self):
        print("Building model...")
        model = Sequential()

        model.add(LSTM(self.hidden_neurons,
                       batch_input_shape=(None, self.timesteps, self.input_dim),
                       return_sequences=True))
        model.add(Dropout(0.2))
        print(model.output_shape) # => (None, 10, 10)

        model.add(LSTM(100, return_sequences=False))
        model.add(Dropout(0.2))
        print(model.output_shape) # => (None, 100)

        model.add(Dense(1))
        print(model.output_shape) # => (None, 1)

        model.add(Activation("linear"))
        print(model.output_shape) # => (None, 1)

        model.compile(loss="mape", optimizer="rmsprop")
        # model.compile(loss='mse', optimizer='rmsprop')

        return model

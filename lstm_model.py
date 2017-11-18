from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.recurrent import LSTM


class LSTMModel:
    def __init__(self):
        self.length_of_sequences = 10
        self.in_out_neurons = 1
        self.hidden_neurons = 10

    def build(self):
        print("Building model...")
        model = Sequential()

        model.add(LSTM(self.hidden_neurons, batch_input_shape=(None, 10, 3), return_sequences=True))
        model.add(Dropout(0.2))
        print(model.output_shape) # => (None, 10, 10)

        model.add(LSTM(100, return_sequences=False))
        model.add(Dropout(0.2))
        print(model.output_shape) # => (None, 100)

        model.add(Dense(1))
        print(model.output_shape) # => (None, 1)

        model.add(Activation("linear"))
        print(model.output_shape) # => (None, 1)

        model.compile(loss="mape", optimizer="adam")
        return model

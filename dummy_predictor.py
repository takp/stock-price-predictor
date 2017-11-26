from sklearn.dummy import DummyClassifier
import numpy as np


class DummyPredictor:
    def __init__(self):
        self.train_size = 807
        self.test_size = 203
        self.features = 3

    def get_evaluation_score(self, y_train):
        # Compare with benchmark model
        dummy_x_train = np.random.random((self.train_size, self.features))
        reshaped_y_train = self.__reshape_y_train(y_train)

        # Build dummy model and predict
        dummy_model = DummyClassifier(strategy="stratified", random_state=0)
        dummy_model.fit(dummy_x_train, reshaped_y_train)
        dummy_x_test = np.array(np.random.random((self.test_size, self.features)))
        return dummy_model.predict(dummy_x_test)

    def __reshape_y_train(self, y_train):
        reshaped_y_train = []
        for i in range(len(y_train)):
            reshaped_y_train.append(y_train[i][0])
        return reshaped_y_train

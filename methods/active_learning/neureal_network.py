import numpy as np
import tensorflow as tf
from keras import Sequential, Input
from keras.layers import Dense, Dropout
from scipy.special import logsumexp
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold

from datasets.dataset_creator import choose_dataset
from methods.utils import set_keras_seed, get_seed


def create_keras_model(layers=[8], act='relu', opt='adam', dr=0.0):
    set_keras_seed(2137)

    model = Sequential()
    model.add(Dense(layers[0], activation=act, input_dim=10))

    for i in range(1, len(layers)):
        model.add(Dense(units=layers[i], activation=act))

    model.add(Dropout(dr))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=['accuracy'])

    return model


# class NeuralNetwork:
#     def __init__(self):
#         self.model = create_keras_model()
#
#     def fit(self, X, y):
#         self.model.fit(X, y, epochs=10, verbose=0)
#         self.model.evaluate(X, y, verbose=0)
#
#     def predict_proba(self, X):
#         # return self.model.predict(X)
#         return np.exp(self.predict_log_proba(X))
#
#     def predict(self, X):
#         return np.argmax(self.model.predict(X), axis=-1)
#
#     def predict_log_proba(self, X):
#         log_prob_x = logsumexp(X, axis=1)
#         return X - np.atleast_2d(log_prob_x).T
#         # return np.log(self.model.predict(X))
#
#     def score(self, X, y):
#         # _, accuracy = self.model.evaluate(X, y, verbose=0)
#         # return accuracy
#         return accuracy_score(y, self.predict(X))

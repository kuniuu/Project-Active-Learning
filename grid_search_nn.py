from keras.wrappers.scikit_learn import KerasClassifier
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold, KFold, train_test_split, GridSearchCV

from datasets.dataset_creator import choose_dataset
from methods.active_learning.neureal_network import create_keras_model
from methods.utils import get_seed

import numpy as np

model = create_keras_model()
print(model.summary())

# Choose a dataset
X_raw, y_raw = choose_dataset('titanic', get_seed())

# Define split
X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.5, random_state=get_seed())

# Fit network
training = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=0)
val_acc = np.mean(training.history['val_accuracy'])
print("\n%s: %.2f%%" % ('val_acc', val_acc * 100))

predictions_probas = model.predict(X_test)
predictions = (predictions_probas > 0.5).astype("int32")
score = accuracy_score(y_test, predictions)
print('Accuracy after prediction: %.5f' % score)


# Plot history for accuracy
# plt.plot(training.history['accuracy'])
# plt.plot(training.history['val_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'validation'], loc='upper left')

# plt.show()

# # Grid search for batch size and epochs
# classifier = KerasClassifier(build_fn=create_keras_model, verbose=0)
#
# batch_size = [16, 32, 64]
# epochs = [10, 50, 100]
# param_map = dict(batch_size=batch_size, epochs=epochs)
#
# grid = GridSearchCV(estimator=classifier, param_grid=param_map, n_jobs=-1, cv=3, verbose=2)
#
# grid_result = grid.fit(X_train, y_train)
#
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print("%f (%f) with: %r" % (mean, stdev, param))
#
# print('\n')
#
# # Grid search for neurons and activation function
# classifier = KerasClassifier(build_fn=create_keras_model, epochs=100, batch_size=64, verbose=0)
#
# layers = [[8], [10], [16], [8, 4], [10, 5], [16, 8], [16, 8, 4], [12, 8, 4]]
# activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
# param_map = dict(layers=layers, act=activation)
#
# grid = GridSearchCV(estimator=classifier, param_grid=param_map, n_jobs=-1, cv=3, verbose=0)
#
# grid_result = grid.fit(X_train, y_train)
#
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print("%f (%f) with: %r" % (mean, stdev, param))
#
# print('\n')

# # Grid search for optimizer
# classifier = KerasClassifier(build_fn=create_keras_model, epochs=100, batch_size=64, verbose=0, layers=[16, 8, 4], act='softsign')
#
# optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Nadam']
# param_map = dict(opt=optimizer)
#
# grid = GridSearchCV(estimator=classifier, param_grid=param_map, n_jobs=-1, cv=3, verbose=2)
#
# grid_result = grid.fit(X_train, y_train)
#
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print("%f (%f) with: %r" % (mean, stdev, param))
#
# print('\n')

# # Grid search for dropout rate
# classifier = KerasClassifier(build_fn=create_keras_model, epochs=100, batch_size=64, layers=[16, 8, 4], act='softsign', verbose=0)
#
# dropout = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.01, 0.05, 0.02]
# param_map = dict(dr=dropout)
#
# grid = GridSearchCV(estimator=classifier, param_grid=param_map, n_jobs=-1, cv=3, verbose=2)
#
# grid_result = grid.fit(X_train, y_train)
#
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print("%f (%f) with: %r" % (mean, stdev, param))
#
# print('\n')

model = create_keras_model(layers=[16, 8, 4], opt='adam', dr=0.0, act='softsign')
print(model.summary())

training = model.fit(X_train, y_train, epochs=100, batch_size=64, validation_split=0.2, verbose=0)
scores = model.evaluate(X_train, y_train)
print("\n%s: %.2f%%" % ('Accuracy while fitting', scores[1] * 100))

plt.plot(training.history['accuracy'])
plt.plot(training.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')

plt.show()

predictions_probas = model.predict(X_test)
predictions = (predictions_probas > 0.5).astype("int32")
score = accuracy_score(y_test, predictions)
print('Accuracy after prediction: %.5f' % score)

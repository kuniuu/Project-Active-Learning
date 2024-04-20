from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
# from scikeras.wrappers import KerasClassifier
from keras.wrappers.scikit_learn import KerasClassifier
# from tensorflow.keras.layers.wrappers.scikit_learn import KerasClassifier

from methods.active_learning.neureal_network import create_keras_model


def choose_estimator(choice, n_neighbors):
    if choice == "GaussianNB":
        estimator = GaussianNB()
    elif choice == "kNearestNeighbours":
        estimator = KNeighborsClassifier(n_neighbors=n_neighbors)
    elif choice == "DecisionTreeClassifier":
        estimator = DecisionTreeClassifier()
    else:
        estimator = KerasClassifier(build_fn=create_keras_model, epochs=1, batch_size=1, layers=[16, 8, 4], act='linear')

    return estimator

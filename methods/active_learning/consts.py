GAUSSIANNB_SYNTHETIC = {
    "estimator": "GaussianNB",
    "dataset": "synthetic"
}

GAUSSIANNB_TITANIC = {
    "estimator": "GaussianNB",
    "dataset": "titanic"
}

KNN_SYNTHETIC = {
    "estimator": "kNearestNeighbours",
    "dataset": "synthetic",
    "n_neighbors": "3"
}

KNN_TITANIC = {
    "estimator": "kNearestNeighbours",
    "dataset": "titanic",
    "n_neighbors": "3"
}

DECISION_TREE_SYNTHETIC = {
    "estimator": "DecisionTreeClassifier",
    "dataset": "synthetic"
}

DECISION_TREE_TITANIC = {
    "estimator": "DecisionTreeClassifier",
    "dataset": "titanic"
}

NEURAL_NETWORK_SYNTHETIC = {
    "estimator": "NeuralNetwork",
    "dataset": "synthetic"
}

NEURAL_NETWORK_TITANIC = {
    "estimator": "NeuralNetwork",
    "dataset": "titanic"
}

ESTIMATORS = [KNN_TITANIC,
              # GAUSSIANNB_TITANIC,
              # KNN_SYNTHETIC,
              # GAUSSIANNB_SYNTHETIC,
              # DECISION_TREE_SYNTHETIC,
              # DECISION_TREE_TITANIC,
              # NEURAL_NETWORK_SYNTHETIC,
              NEURAL_NETWORK_TITANIC]

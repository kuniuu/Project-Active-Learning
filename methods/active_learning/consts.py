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

ESTIMATORS = [GAUSSIANNB_SYNTHETIC,
              GAUSSIANNB_TITANIC,
              KNN_SYNTHETIC,
              KNN_TITANIC,
              DECISION_TREE_SYNTHETIC,
              DECISION_TREE_TITANIC]


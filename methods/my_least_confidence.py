import numpy as np
from modAL.utils import multi_argmax
from modAL.utils.data import modALinput
from sklearn.base import BaseEstimator


def least_confidence_sampling(classifier, X, n_instances: int = 1):
    y = classifier.predict_proba(X)
    uncertainty = 1 - np.max(y, axis=1)
    return multi_argmax(uncertainty, n_instances=n_instances)

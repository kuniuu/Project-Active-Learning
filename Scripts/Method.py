from sklearn.base import BaseEstimator
import numpy as np

class UncertaintySampling(BaseEstimator):
    def __init__(self, model, budget):
        self.model = model
        self.budget = budget

    def fit(self, X, y):
        self.X_ = X
        self.y_ = y
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X=None, y=None):
        if X is None:
            X = self.X_
        if y is None:
            y = self.y_
        return np.mean(self.predict(X) == y)

    def query(self, X_pool):
        uncertainty = -self.model.predict_proba(X_pool) * np.log2(self.model.predict_proba(X_pool))
        uncertainty = uncertainty.sum(axis=1)
        query_idx = np.argpartition(uncertainty, -self.budget)[-self.budget:]
        return query_idx, X_pool[query_idx]